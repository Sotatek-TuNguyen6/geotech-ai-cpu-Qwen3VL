import os, json, time, threading
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
    TextIteratorStreamer,
)

MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen3-VL-8B-Instruct")
PROMPT = os.getenv("PROMPT", "Describe this image in detail.")
LIMIT = int(os.getenv("LIMIT", "100"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "256"))
OMP_THREADS = int(os.getenv("OMP_NUM_THREADS", "1"))

IN_DIR = Path(os.getenv("IN_DIR", "/data/in"))
OUT_PATH = Path(os.getenv("OUT_PATH", "/data/out/results.jsonl"))

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

def main():
    os.environ["OMP_NUM_THREADS"] = str(OMP_THREADS)
    torch.set_num_threads(OMP_THREADS)

    device = torch.device("cpu")

    print(f"[INFO] Loading model: {MODEL_ID}", flush=True)
    print(f"[INFO] Device: {device}", flush=True)
    
    print("[INFO] Loading processor...", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    print("[INFO] Processor loaded!", flush=True)
    
    print("[INFO] Loading model weights (this may take several minutes on CPU)...", flush=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device).eval()
    print("[INFO] Model loaded successfully!", flush=True)

    # paths = sorted(
    #     [p for p in IN_DIR.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    # )[:LIMIT]
    # if not paths:
    #     raise SystemExit("No images found")
    
    paths = [Path("CKK20241-C1-41.jpg")] * 2

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    total_start = time.perf_counter()

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for p in tqdm(paths, desc="processing"):
            img = Image.open(p).convert("RGB")
            img.thumbnail((1024, 1024))

            messages = [{
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": PROMPT}],
            }]
            prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True)

            inputs = processor(text=[prompt_text], images=[img], return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Streamer để bắt token đầu tiên
            streamer = TextIteratorStreamer(
                processor.tokenizer,
                skip_special_tokens=True,
                skip_prompt=True
            )

            # Chạy generate ở thread riêng để main thread đọc streamer
            gen_kwargs = dict(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                streamer=streamer,
            )

            start = time.perf_counter()
            t = threading.Thread(target=model.generate, kwargs=gen_kwargs)
            t.start()

            # Đọc token stream để đo TTFT
            ttft = None
            collected = []

            for piece in streamer:
                if ttft is None:
                    ttft = time.perf_counter() - start
                collected.append(piece)

            t.join()
            total = time.perf_counter() - start

            text = "".join(collected).strip()
            if ttft is None:
                ttft = total  # trường hợp hiếm không ra token

            # Ước lượng token output (không cần gọi generate lại)
            # (đếm token từ text; không hoàn hảo 100% nhưng đủ cho profiling)
            tokens = len(processor.tokenizer.encode(text, add_special_tokens=False))

            rec = {
                "file": p.name,
                "ttft_seconds": round(ttft, 3),
                "total_seconds": round(total, 3),
                "tokens": tokens,
                "text": text
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    total_elapsed = time.perf_counter() - total_start
    avg = total_elapsed / len(paths)

    print(f"Done {len(paths)} images")
    print(f"Wall time: {total_elapsed:.2f}s")
    print(f"Avg wall / image: {avg:.2f}s")
    print(f"Output: {OUT_PATH}")

if __name__ == "__main__":
    main()
