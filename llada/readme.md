# Fast-dLLM with LLaDA Inference

```bash
python infer_llada_poisson.py \
  --model_path GSAI-ML/LLaDA-8B-Instruct \
  --split test \
  --limit 10 \
  --arrival_rate 2.0 \
  --max_batch_size 4 \
  --steps 256 \
  --max_batch_wait_ms 0 \
  --gen_length 256 \
  --block_length 32 \
  --out_jsonl outs/preds_poisson_gsm8k.json
