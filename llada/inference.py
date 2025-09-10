#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, time, os, random, threading, queue
import torch
import numpy as np
import torch.nn.functional as F

from typing import List, Dict, Any, Optional
from transformers import AutoConfig, AutoTokenizer
from datasets import load_dataset
from model.modeling_llada import LLaDAModelLM

MASK_ID = 126336  # LLaDA [MASK]

def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(24)
    torch.set_num_interop_threads(24)

def set_global_determinism(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# ---------------------------
# Producer
# ---------------------------
def producer_thread(
    data: List[Dict[str, Any]],
    q: "queue.Queue[Dict[str, Any]]",
    arrival_rate: float,
    start_idx: int,
    max_requests: int,
    stop_event: threading.Event,
    seed: int
):
    lam = float(arrival_rate)
    assert lam > 0.0, "--arrival_rate must be > 0"
    produced = 0
    i = start_idx
    rng = random.Random(seed)
    while not stop_event.is_set() and produced < max_requests and i < len(data):
        dt = rng.expovariate(lam)
        if stop_event.wait(dt):
            break
        now = time.perf_counter()
        q.put({"idx": i, "example": data[i], "arrival_time": now})
        produced += 1
        i += 1
    q.put({"_done": True})

# ---------------------------
# Data loader (GSM8K)
# ---------------------------
def load_gsm8k(split: str, limit: Optional[int], fewshot_k: int, seed: int) -> List[Dict[str, Any]]:
    ds = load_dataset("gsm8k", "main", split=split)
    exemplars: List[Dict[str, str]] = []
    if fewshot_k > 0:
        train = load_dataset("gsm8k", "main", split="train")
        rnd = random.Random(seed)
        idxs = rnd.sample(range(len(train)), k=min(fewshot_k, len(train)))
        for i in idxs:
            q = train[i]["question"].strip()
            a = train[i]["answer"].strip()
            exemplars.append({"q": q, "a": a})

    fewshot_prefix = ""
    if exemplars:
        shots = []
        for ex in exemplars:
            shots.append(f"Question: {ex['q']}\nAnswer: {ex['a']}\n")
        fewshot_prefix = "\n".join(shots).strip() + "\n\n"

    out: List[Dict[str, Any]] = []
    n = len(ds) if not limit or limit <= 0 else min(limit, len(ds))
    for i in range(n):
        q = ds[i]["question"].strip()
        user_prompt = (
            f"{fewshot_prefix}"
            f"Question: {q}\n"
            "Please solve step by step. On the last line, write the final answer as '#### <number>'."
        )
        out.append({"prompt": user_prompt, "stop": []})
    return out

# ---------------------------
# Collate
# ---------------------------
def collate_batch(tok: AutoTokenizer, batch: List[Dict[str, Any]], is_instruct: bool, device: torch.device):
    input_ids_list: List[List[int]] = []
    stop_ids_per_sample: List[List[List[int]]] = []
    lens: List[int] = []

    for ex in batch:
        text = ex["prompt"]
        if is_instruct:
            msg = [{"role": "user", "content": text}]
            rendered = tok.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
        else:
            rendered = text
        ids = tok(rendered)["input_ids"]
        input_ids_list.append(ids)
        lens.append(len(ids))

        stops = ex.get("stop") or []
        stop_ids_per_sample.append([tok(s, add_special_tokens=False)["input_ids"] for s in stops])

    max_len = max(lens)
    rows = []

    # 가장 긴 request의 토큰 길이를 기준으로 나머지는 패딩을 붙임
    for ids in input_ids_list:
        pad = [tok.pad_token_id] * (max_len - len(ids))
        rows.append(torch.tensor(pad + ids, dtype=torch.long, device=device).unsqueeze(0))
    input_ids = torch.cat(rows, dim=0)
    return {"input_ids": input_ids, "prompt_width": max_len, "stop_ids_batch": stop_ids_per_sample}

def count_generated_tokens_1d(
    ids_1d: torch.Tensor,
    pad_id: Optional[int],
    mask_id: Optional[int],
    eos_id: Optional[int],
) -> int:
    """
    주어진 1D 토큰 시퀀스(ids_1d)에서
    - EOS 이후 토큰은 무시하고
    - PAD 및 MASK 토큰은 제외한
    실제 생성된 토큰 개수를 카운트한다.
    """

    # 1) EOS 토큰이 있으면, 그 위치까지만 고려 (EOS 이후는 잘라냄)
    if eos_id is not None:
        eos_pos = (ids_1d == eos_id).nonzero(as_tuple=False)
        if eos_pos.numel() > 0:
            # 첫 번째 EOS 위치까지만 사용
            ids_1d = ids_1d[: int(eos_pos[0].item())]

    # 2) 모든 토큰을 "유효(true)"로 시작
    valid = torch.ones_like(ids_1d, dtype=torch.bool)

    # 3) PAD 토큰 제외
    if pad_id is not None:
        valid &= (ids_1d != pad_id)
    
    # 4) MASK 토큰 제외
    if mask_id is not None:
        valid &= (ids_1d != mask_id)

    # 5) 유효(true) 토큰 개수를 세어 반환
    return int(valid.sum().item())

def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    # base: 각 step에서 최소한 채워야 하는 토큰 수 (균등 분배 몫)
    # remainder: 균등 분배 후 남는 나머지 (step마다 1개씩 추가로 분배할 때 씀)
    base = mask_num // steps
    remainder = mask_num % steps

    # 처음에는 모든 step마다 base 개수만큼 할당.
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    # 남은 remainder만큼 step 앞쪽부터 1개씩 더 줌.
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens

def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise

def get_transfer_index(logits, temperature, remasking, mask_index, x, num_transfer_tokens, threshold=None):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)
    
    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    if threshold is not None:
        num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    for j in range(confidence.shape[0]):
        _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j])
        transfer_index[j, select_index] = True
        if threshold is not None:
            for k in range(1, num_transfer_tokens[j]):
                if confidence[j, select_index[k]] < threshold:
                    transfer_index[j, select_index[k]] = False
    return x0, transfer_index

@ torch.no_grad()
def generate_with_dual_cache(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
            remasking='low_confidence', mask_id=126336, threshold=None, factor=None,
            eos_token_id=None,
            stop_token_ids=None,   # list[list[int]] or None
            event_cb=None):        # callable(step:int, y:LongTensor[B,T])
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''

    """
    블록 디퓨전(block diffusion) 방식으로 연속 생성:
      - 입력 prompt 뒤에 길이 gen_length의 [MASK] 영역을 붙여 시작.
      - gen_length를 block_length로 나눈 '블록' 단위로 생성/수정.
      - 각 블록 내부에서는 여러 업데이트 스텝을 반복하며 [MASK]를 점진적으로 실 토큰으로 채움.
      - 듀얼 캐시(dual cache) 방식: 처음 full-seq로 KV 캐시를 만들고, 이후 블록 슬라이스만 재사용하며 업데이트.

    핵심 아이디어:
      1) 전체 생성 길이(gen_length)를 block_length씩 나눈 구간(블록)을 순서대로 처리
      2) 각 블록마다 "남은 마스크 수"에 비례해 step 당 채울 개수(num_transfer_tokens)를 배분
      3) 첫 full forward로 past_key_values(캐시) 초기화 후, 해당 블록 슬라이스만 캐시 재사용하며 반복 업데이트
      4) remasking 규칙에 따라 신뢰도 낮은 위치를 중심으로 [MASK] -> 예측토큰 전환
      5) (옵션) 매 블록 첫 업데이트 직후와 이후 반복 업데이트마다 event_cb로 중간 결과 알림
    """

    # x = [ 프롬프트 | 생성영역(MASK로 초기화) ]  형태의 입력 텐서 준비
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    # 블록 개수와 블록당 스텝 수 산출(정확히 나누어떨어진다는 가정)
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    """
    # 전체 샘플링 스텝 수(steps)를 블록 단위로 균등 분배
    # Example: gen_length=128, block_length=32 → num_blocks = 128/32 = 4
    steps=128 → 블록마다 128 // 4 = 32 스텝씩 할당
    즉, 전체 128 step이 4개의 블록에 균등하게 분배됩니다.
    Block 1-4: 각 32 step씩 수행
    """
    assert steps % num_blocks == 0
    steps = steps // num_blocks

    nfe = 0  # number of model forward evaluations(계산량/호출 횟수 추정용 카운터)

    # === 블록 루프: 생성영역을 block_length씩 순차적으로 채움 ===
    for num_block in range(num_blocks):
        # 현재 블록의 절대 인덱스 범위
        current_block_start = prompt.shape[1] + num_block * block_length
        current_block_end = current_block_start + block_length

        # 현재 블록 내 아직 [MASK]인 위치(초기에는 전부 MASK)
        block_mask_index = (x[:, current_block_start:current_block_end] == mask_id)

        # 블록 내부에서 step마다 채울 토큰 수(배치별로 다를 수 있음) 계산
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        # 전체 x에 대해 1회 forward: past_key_values(=KV 캐시) 생성
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t0 = time.perf_counter()
        output = model(x, use_cache=True)   # 캐시 생성 경로
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t1 = time.perf_counter()
        # print(f"[block {num_block}] first forward time (w/o kv cache): {t1 - t0:.6f}s")

        past_key_values = output.past_key_values
        nfe += 1

        # 이 블록 이후의 영역은 이번 블록에서 건드리지 않도록 마스크 범위를 제한
        mask_index = (x == mask_id)
        mask_index[:, current_block_end:] = 0

        # 리마스킹 정책에 따라 이번 스텝에서 실제로 채울 위치(transfer_index)와 값(x0)을 고름
        #  - threshold: low_confidence에서 신뢰도 컷오프에 사용 가능
        #  - factor: 동적 리마스킹에서 사용
        if factor is None:
            x0, transfer_index = get_transfer_index(
                output.logits, 
                temperature, 
                remasking, 
                mask_index, 
                x, 
                num_transfer_tokens[:, 0] if threshold is None else None, 
                threshold
            )
        else:
            x0, transfer_index = get_transfer_index_dynamic(
                output.logits,
                temperature,
                remasking,
                mask_index,
                x,
                None,
                factor
            )

        x[transfer_index] = x0[transfer_index]

        # --- ADD: callback after the first update of this block ---
        # 지금은 활용 X
        if event_cb is not None:
            # torch.cuda.synchronize()   # 필요 시 활성화
            step_idx = num_block * steps + 0  # 블록 첫 업데이트
            event_cb(step_idx, x)

        i = 1 # === 블록 내 반복 업데이트 루프 ===
        # replace_position: 모델에 "이번엔 이 구간을 교체/업데이트한다"는 힌트를 주는 마스크
        replace_position = torch.zeros_like(x, dtype=torch.bool)
        replace_position[:, current_block_start:current_block_end] = 1
        while True:
            # 더 이상 이 블록 안에 MASK가 없으면 종료
            if (x[:, current_block_start:current_block_end] == mask_id).sum() == 0:
                break

            nfe += 1

            # 현재 블록에서 아직 MASK인 위치
            mask_index = (x[:, current_block_start:current_block_end] == mask_id)

            # 캐시를 재사용하며, 블록 슬라이스만 forward
            #  - 입력은 x의 [current_block_start:current_block_end] 슬라이스
            #  - replace_position을 넘겨 "절대위치/캐시 정렬"을 맞춰줌
            if torch.cuda.is_available(): torch.cuda.synchronize()
            t0 = time.perf_counter()
            logits = model(
                x[:, current_block_start:current_block_end],
                past_key_values=past_key_values,
                use_cache=True,
                replace_position=replace_position
            ).logits
            if torch.cuda.is_available(): torch.cuda.synchronize()
            t1 = time.perf_counter()
            # print(f"[block {num_block}] decode forward time (w/ kv cache): {t1 - t0:.6f}s")
            # print(f"[block {num_block} step {i}] decode with past: {t1 - t0:.6f}s")

            # 이번 반복에서 채울 위치/개수 산정(정책·온도·임계치 등에 따라)
            if factor is None:
                x0, transfer_index = get_transfer_index(
                    logits,
                    temperature,
                    remasking,
                    mask_index,
                    x[:, current_block_start:current_block_end],
                    num_transfer_tokens[:, i] if threshold is None else None,
                    threshold
                )
            else:
                x0, transfer_index = get_transfer_index_dynamic(
                    logits,
                    temperature,
                    remasking,
                    mask_index,
                    x[:, current_block_start:current_block_end],
                    None,
                    factor
                )

            # 슬라이스에 직접 덮어쓰기: MASK -> 예측토큰
            x[:, current_block_start:current_block_end][transfer_index] = x0[transfer_index]

            # --- ADD: callback after every update inside while-loop ---
            if event_cb is not None:
                # torch.cuda.synchronize()
                step_idx = num_block * steps + (i - 1)   # 이번 갱신 시점
                event_cb(step_idx, x)

            i += 1

    return x, nfe

# ==============================
# Main
# ==============================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--limit", type=int, default=200, help="GSM8K 샘플 상한(프로듀서가 생성)")
    ap.add_argument("--fewshot_k", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1234)

    # Poisson/Batch policy
    ap.add_argument("--arrival_rate", type=float, default=2.0, help="λ (req/s)")
    ap.add_argument("--max_batch_size", type=int, default=4)
    ap.add_argument("--max_batch_wait_ms", type=int, default=50, help="배치 타임아웃(ms)")

    # Generation params
    ap.add_argument("--gen_length", type=int, default=256)
    ap.add_argument("--block_length", type=int, default=32)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=0.0)

    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out_jsonl", type=str, default="outs/preds_poisson_gsm8k.jsonl")
    args = ap.parse_args()

    # Fix seed to match experimental results
    set_seed(args.seed)
    set_global_determinism(1234)

    device = torch.device(args.device)
    config = AutoConfig.from_pretrained(args.model_path)
    config.flash_attention = False # 공통적으로 꺼두고 실험
    model = LLaDAModelLM.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, config=config
    ).to(device).eval()

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    is_instruct = "instruct" in (args.model_path or "").lower()

    data = load_gsm8k(args.split, args.limit, args.fewshot_k, args.seed)
    total_target = min(args.limit, len(data)) if args.limit and args.limit > 0 else len(data)
    
    os.makedirs(os.path.dirname(args.out_jsonl) or ".", exist_ok=True)
    fout = open(args.out_jsonl, "w", encoding="utf-8")

    steps = args.steps
    total_tokens = 0
    q_in: "queue.Queue[Dict[str, Any]]" = queue.Queue()
    stop_evt = threading.Event()
    prod = threading.Thread(
        target=producer_thread,
        kwargs=dict(
            data=data,
            q=q_in,
            arrival_rate=args.arrival_rate,
            start_idx=0,
            max_requests=total_target,
            stop_event=stop_evt,
            seed=args.seed,
        ),
        daemon=True,
    )

    t_zero = time.perf_counter()
    prod.start()

    pending: List[Dict[str, Any]] = []
    seen_done_marker = False
    processed = 0

    last_batch_open_t: Optional[float] = None
    max_wait_s = args.max_batch_wait_ms / 1000.0

    print(f"[INFO] start serving with Poisson arrivals λ={args.arrival_rate} req/s, "
          f"max_batch_size={args.max_batch_size}, batch_wait={args.max_batch_wait_ms}ms")

    while processed < total_target or not seen_done_marker or pending:
        # absorb arrivals
        try:
            item = q_in.get(timeout=0.01)
            if "_done" in item:
                seen_done_marker = True
            else:
                pending.append(item)
                if last_batch_open_t is None:
                    last_batch_open_t = time.perf_counter()
        except queue.Empty:
            pass

        # flush condition
        now = time.perf_counter()
        should_flush = False
        if pending:
            '''
            다음과 같은 세 가지 경우에 batch를 실행함
            '''
            # 1) batch 가능한 충분한 request 존재할 경우  
            if len(pending) >= args.max_batch_size:
                should_flush = True
            # 2) 마지막으로 request가 제출된 시점으로부터 max_wait_s 이상 흐른 경우
            elif last_batch_open_t is not None and (now - last_batch_open_t) >= max_wait_s:
                should_flush = True
            # 3) 이미 모든 request가 제출되었을 경우
            elif seen_done_marker and processed + len(pending) >= total_target:
                should_flush = True

        if not should_flush:
            continue

        # form batch
        batch = pending[:args.max_batch_size]
        pending = pending[args.max_batch_size:]
        last_batch_open_t = time.perf_counter() if pending else None

        ex_list = [b["example"] for b in batch]
        # batch 할 prompt들을 하나로 합침
        pack = collate_batch(tok, ex_list, is_instruct, device)
        input_ids = pack["input_ids"]
        prompt_width = pack["prompt_width"]
        stop_ids_batch = pack["stop_ids_batch"]
        batch_arrivals = [b["arrival_time"] for b in batch]

        # timing (aligned fields)
        # if torch.cuda.is_available(): torch.cuda.synchronize()

        service_start = time.perf_counter()  # flush/decoding start
        y, nfe = generate_with_dual_cache(
            model, input_ids,
            steps=steps, gen_length=args.gen_length, block_length=args.block_length,
            temperature=args.temperature, remasking="low_confidence", mask_id=MASK_ID,
            threshold=None, factor=None,
            eos_token_id=tok.eos_token_id, stop_token_ids=stop_ids_batch, event_cb=None
        )

        # if torch.cuda.is_available(): torch.cuda.synchronize()
        finish_time = time.perf_counter()    # flush/decoding end

        # write results
        B = y.size(0)
        for b in range(B):
            idx = batch[b]["idx"]
            gen_ids = y[b, prompt_width:]
            raw = tok.decode(gen_ids, skip_special_tokens=False)

            for st in (ex_list[b].get("stop") or []):
                if st and st in raw:
                    raw = raw.split(st)[0]

            text = tok.decode(tok(raw)["input_ids"], skip_special_tokens=True)
            gen_tok = count_generated_tokens_1d(
                gen_ids,
                pad_id=tok.pad_token_id,
                mask_id=MASK_ID,
                eos_id=tok.eos_token_id,
            )
            total_tokens += gen_tok

            arrival = batch_arrivals[b]
            rec = {
                "index": idx,
                "arrival_time": arrival,
                "service_start": service_start,
                "finish_time": finish_time,
                "queue_wait_s": service_start - arrival,
                "service_time_s": finish_time - service_start,
                "sojourn_time_s": finish_time - arrival,
                "gen_tokens": gen_tok,
                "nfe": int(nfe),
                "prompt": ex_list[b]["prompt"],
                "prediction": text,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            processed += 1

    wall = time.perf_counter() - t_zero
    print("\n====== Summary ======")
    print(f"Produced & processed: {processed}/{total_target}")
    print(f"Total wall time: {wall:.3f}s")
    print(f"Approx tokens generated: {total_tokens}")
    if wall > 0:
        print(f"Throughput: {total_tokens / wall:.2f} tokens/s")

if __name__ == "__main__":
    main()
