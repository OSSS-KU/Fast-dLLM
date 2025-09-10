#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, time, os, random, threading, queue
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoConfig, AutoTokenizer
from datasets import load_dataset
from model.modeling_llada_selective_batching import LLaDAModelLM

MASK_ID = 126336  # LLaDA [MASK]


# ---------------------------
# Utilities
# ---------------------------
def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


# ---------------------------
# Data loader (GSM8K minimal)
# ---------------------------
def load_gsm8k(split: str, limit: Optional[int], fewshot_k: int, seed: int) -> List[Dict[str, Any]]:
    ds = load_dataset("gsm8k", "main", split=split)
    exemplars: List[Dict[str, str]] = []
    if fewshot_k > 0:
        train = load_dataset("gsm8k", "main", split="train")
        rnd = random.Random(seed)
        idxs = rnd.sample(range(len(train)), k=min(fewshot_k, len(train)))
        for i in idxs:
            exemplars.append({"q": train[i]["question"].strip(), "a": train[i]["answer"].strip()})
    fewshot_prefix = ""
    if exemplars:
        fewshot_prefix = "\n".join([f"Question: {ex['q']}\nAnswer: {ex['a']}\n" for ex in exemplars]).strip() + "\n\n"
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
# Producer (Poisson arrivals)
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
    lam = float(arrival_rate); assert lam > 0.0
    produced = 0; i = start_idx; rng = random.Random(seed)
    while not stop_event.is_set() and produced < max_requests and i < len(data):
        dt = rng.expovariate(lam)
        if stop_event.wait(dt): break
        now = time.perf_counter()
        q.put({"idx": i, "example": data[i], "arrival_time": now})
        produced += 1; i += 1
    q.put({"_done": True})


# ---------------------------
# Token count helper
# ---------------------------
def count_generated_tokens_1d(ids_1d: torch.Tensor, pad_id: Optional[int], mask_id: Optional[int], eos_id: Optional[int]) -> int:
    if eos_id is not None:
        eos_pos = (ids_1d == eos_id).nonzero(as_tuple=False)
        if eos_pos.numel() > 0:
            ids_1d = ids_1d[: int(eos_pos[0].item())]
    valid = torch.ones_like(ids_1d, dtype=torch.bool)
    if pad_id is not None: valid &= (ids_1d != pad_id)
    if mask_id is not None: valid &= (ids_1d != mask_id)
    return int(valid.sum().item())

def add_gumbel_noise(logits: torch.Tensor, temperature: float):
    """
    logits에 Gumbel noise를 추가해 샘플링 효과를 줌.
    temperature=0이면 그대로 반환.
    """
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel = -torch.log(-torch.log(noise + 1e-9) + 1e-9)
    return logits + gumbel * temperature


def get_transfer_index(
    logits: torch.Tensor,            # [1, L, V]
    temperature: float,
    remasking: str,
    mask_index: torch.Tensor,        # [1, L]
    x: torch.Tensor,           # [1, L]
    num_transfer_tokens: torch.Tensor, # [1] or [1,1]
    threshold: Optional[float] = None,
):
    """
    logits와 마스크 정보를 이용해,
    이번 step에서 어떤 토큰을 치환할지 결정.
    반환: (x0, transfer_index)
      - x0: 예측 토큰 (shape [1, L])
      - transfer_index: True/False 마스크 (shape [1, L])
    """
    if logits.dim() == 2:
        logits = logits.unsqueeze(0)  # [1, L, V]

    B, L, V = logits.shape

    # 1. Gumbel noise 추가 후 argmax
    logits_noise = add_gumbel_noise(logits, temperature)
    x0 = torch.argmax(logits_noise, dim=-1)  # [B, L]

    # 2. confidence 계산
    if remasking == "low_confidence":
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)  # [B, L]
    elif remasking == "random":
        x0_p = torch.rand((B, L), device=logits.device)
    else:
        raise NotImplementedError(remasking)

    # 3. 원래 토큰 유지
    x0 = torch.where(mask_index, x0, x)

    # 4. confidence 마스크
    conf = torch.where(mask_index, x0_p, torch.full_like(x0_p, -float("inf")))

    # 5. top-k로 선택
    transfer_index = torch.zeros_like(x0, dtype=torch.bool)
    for j in range(B):
        k = int(num_transfer_tokens[j].item())
        if k > 0:
            _, sel = torch.topk(conf[j], k=k)
            transfer_index[j, sel] = True

    return x0, transfer_index

# ---------------------------
# Request state (dLLM)
# ---------------------------
class ReqState:
    def __init__(self, rid: int, prompt_ids: List[int], prompt_w: int,
                 gen_len: int, block_len: int, steps_per_block: int, device: torch.device):
        self.id = rid
        self.prompt_w = prompt_w
        self.gen_len = gen_len
        self.block_len = block_len
        self.steps_per_block = steps_per_block

        T = prompt_w + gen_len
        self.x = torch.full((1, T), MASK_ID, dtype=torch.long, device=device)
        self.x[:, :prompt_w] = torch.tensor(prompt_ids, device=device).view(1, -1)

        self.block_idx = 0
        self.step_in_block = 0
        self.prefill_done = False
        self.done = False
        self.kv_handle = None
        self._num_transfer_table = None  # [1, steps_per_block]

    def current_block_span(self) -> Tuple[int, int]:
        s = self.prompt_w + self.block_idx * self.block_len
        e = min(self.prompt_w + self.gen_len, s + self.block_len)
        return s, e


def build_batch(active: List[ReqState], max_batch_size: int):
    tokens_list, pos_list, rid_list = [], [], []
    for r in active:
        if r.done:
            continue
        if len(rid_list) >= max_batch_size:
            break

        if not r.prefill_done:
            s, e = 0, r.prompt_w
        else:
            s, e = r.current_block_span()

        toks = r.x[:, s:e].view(-1)  # [Li]
        pos = torch.arange(s, e, device=r.x.device, dtype=torch.long)
        tokens_list.append(toks)
        pos_list.append(pos)
        rid_list.append(r.id)

    if not tokens_list:
        return None, None, []

    tokens_batch = torch.cat(tokens_list, dim=0)  # [Σ Li]
    pos_batch = torch.cat(pos_list, dim=0)        # [Σ Li]
    return tokens_batch, pos_batch, rid_list

    # 2) padding 해서 [B, Lmax] 형태로 맞춤
    B = len(tokens_list)
    tokens_batch = torch.full((B, max_len), MASK_ID, dtype=torch.long, device=tokens_list[0].device)
    pos_batch = torch.zeros((B, max_len), dtype=torch.long, device=pos_list[0].device)

    for i in range(B):
        L_i = tokens_list[i].size(0)
        tokens_batch[i, :L_i] = tokens_list[i]
        pos_batch[i, :L_i] = pos_list[i]

    return tokens_batch, pos_batch, rid_list

def get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    """
    각 request의 MASK 토큰 개수를 steps 단계에 균등 분배.
    mask_index: [1, L] (True = 아직 MASK 상태)
    steps: 블록 내 step 수
    return: [1, steps] → 각 step에서 채워야 하는 토큰 수
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)  # [1,1]
    base = mask_num // steps
    rem = mask_num % steps

    out = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        out[i, :rem[i]] += 1
    return out

# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--fewshot_k", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1234)

    # Arrivals / batching
    ap.add_argument("--arrival_rate", type=float, default=2.0, help="λ (req/s)")
    ap.add_argument("--max_batch_size", type=int, default=4, help="동시에 묶어서 forward할 요청 수")
    ap.add_argument("--max_batch_wait_ms", type=int, default=50)

    # dLLM params
    ap.add_argument("--gen_length", type=int, default=256)
    ap.add_argument("--block_length", type=int, default=32)
    ap.add_argument("--steps_per_block", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--remasking", type=str, default="low_confidence", choices=["low_confidence", "random"])

    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out_jsonl", type=str, default="outs/preds_iter_sched.jsonl")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    config = AutoConfig.from_pretrained(args.model_path)
    config.flash_attention = False
    model = LLaDAModelLM.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, config=config
    ).to(device).eval()

    tok = AutoTokenizer.from_pretrained(args.model_path)
    if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id

    data = load_gsm8k(args.split, args.limit, args.fewshot_k, args.seed)
    total_target = min(args.limit, len(data)) if args.limit and args.limit > 0 else len(data)

    os.makedirs(os.path.dirname(args.out_jsonl) or ".", exist_ok=True)
    fout = open(args.out_jsonl, "w", encoding="utf-8")

    # ---- Arrivals
    q_in: "queue.Queue[Dict[str, Any]]" = queue.Queue()
    stop_evt = threading.Event()
    prod = threading.Thread(
        target=producer_thread,
        kwargs=dict(
            data=data, q=q_in, arrival_rate=args.arrival_rate,
            start_idx=0, max_requests=total_target, stop_event=stop_evt, seed=args.seed
        ),
        daemon=True,
    )
    prod.start()

    # ---- Scheduler state
    pending: List[Dict[str, Any]] = []
    active: List[ReqState] = []
    reqid_to_state: Dict[int, ReqState] = {}
    next_req_id = 0
    seen_done_marker = False
    processed = 0
    total_tokens = 0

    max_wait_s = args.max_batch_wait_ms / 1000.0
    last_open_t: Optional[float] = None

    t0 = time.perf_counter()
    print(f"[INFO] iterative scheduling, λ={args.arrival_rate}, max_batch_size={args.max_batch_size}")

    while processed < total_target or not seen_done_marker or pending or any(not r.done for r in active):
        # 1) absorb new arrivals
        try:
            item = q_in.get(timeout=0.001)
            if "_done" in item:
                seen_done_marker = True
            else:
                pending.append(item)
                if last_open_t is None: last_open_t = time.perf_counter()
        except queue.Empty:
            pass

        # 2) admission
        while pending and len(active) < args.max_batch_size:
            it = pending.pop(0)
            idx = it["idx"]; ex = it["example"]
            ids = tok(ex["prompt"])["input_ids"]
            r = ReqState(
                rid=next_req_id, prompt_ids=ids, prompt_w=len(ids),
                gen_len=args.gen_length, block_len=args.block_length,
                steps_per_block=args.steps_per_block, device=device
            )
            r.arrival_time = it["arrival_time"]
            r.index = idx
            r.stop_strs = ex.get("stop") or []
            active.append(r); reqid_to_state[r.id] = r
            next_req_id += 1

        # 3) build batch
        tokens_batch, pos_batch, rid_list = build_batch(active, args.max_batch_size)
        if not rid_list:
            time.sleep(0.001)
            continue

        # 4) forward (모델 내부 구현 필요)
        outputs = model.forward_batch(
            tokens_batch.to(device),
            pos_batch.to(device),
            rid_list
        )

        # === 이후 부분은 요청 상태 업데이트 + 결과 flush ===
        logits_cat: torch.Tensor = outputs["logits_cat"]  # [ΣL, V]

        cursor = 0
        still_active: List[ReqState] = []
        for rid in rid_list:
            r = reqid_to_state[rid]
            if r.done:
                continue

            if not r.prefill_done:
                s, e = 0, r.prompt_w
                r.prefill_done = True
            else:
                s, e = r.current_block_span()

            L_req = e - s
            logits_slice = logits_cat[cursor:cursor + L_req]   # [L_req, V]
            cursor += L_req

            mask_index = (r.x[:, s:e] == MASK_ID)

            if r._num_transfer_table is None:
                r._num_transfer_table = get_num_transfer_tokens(
                    mask_index, r.steps_per_block
                )

            step_idx = r.step_in_block
            num_transfer_tokens = r._num_transfer_table[:, step_idx]

            x0, transfer_index = get_transfer_index(
                logits_slice.unsqueeze(0),
                temperature=args.temperature,
                remasking=args.remasking,
                mask_index=mask_index,
                x=r.x[:, s:e],
                num_transfer_tokens=num_transfer_tokens,
                threshold=None
            )

            # === 수정된 부분 ===
            mask_index_1d = transfer_index[0]   # [L_req]
            r.x[0, s:e][mask_index_1d] = x0[0, mask_index_1d]
            # r.x[:, s:e][transfer_index] = x0[0, transfer_index]

            r.step_in_block += 1
            if r.step_in_block >= r.steps_per_block:
                r.block_idx += 1
                r.step_in_block = 0
                r._num_transfer_table = None

            # 완료 조건 체크
            mask_remain = (r.x[:, r.prompt_w:] == MASK_ID).sum().item()
            if r.block_idx * r.block_len >= r.gen_len or mask_remain == 0:
                r.done = True
                gen_ids = r.x[:, r.prompt_w:]
                text = tok.decode(gen_ids[0], skip_special_tokens=True)
                gen_tok = count_generated_tokens_1d(
                    gen_ids[0],
                    pad_id=tok.pad_token_id,
                    mask_id=MASK_ID,
                    eos_id=tok.eos_token_id,
                )
                total_tokens += gen_tok

                rec = {
                    "index": r.id,
                    "arrival_time": getattr(r, "arrival_time", None),
                    "finish_time": time.perf_counter(),
                    "gen_tokens": gen_tok,
                    "prediction": text,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                processed += 1
            else:
                still_active.append(r)

        active = still_active

    wall = time.perf_counter() - t0
    fout.close()
    print("\n====== Summary ======")
    print(f"Processed: {processed}/{total_target}")
    print(f"Total wall time: {wall:.3f}s")
    if wall > 0:
        print(f"Throughput: {total_tokens / wall:.2f} tokens/s")


if __name__ == "__main__":
    main()
