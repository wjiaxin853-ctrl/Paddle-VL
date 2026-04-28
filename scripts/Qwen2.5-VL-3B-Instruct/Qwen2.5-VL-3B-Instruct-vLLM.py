#!/usr/bin/env python3

from __future__ import annotations

import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from PIL import Image
from vllm import LLM, SamplingParams

MODEL_PATH = "/workspace/models/Qwen2.5-VL-3B-Instruct"
IMAGE_PATH = "/workspace/demo/images/手持身份证.jpg"
OUTPUT_DIR = "/workspace/output/Qwen2.5-VL-3B-Instruct-vLLM"
PROMPT = "请描述这张图片的主要内容，并尽量提取其中清晰可见的文字。"


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    image = Image.open(IMAGE_PATH).convert("RGB")

    llm = LLM(
        model=MODEL_PATH,
        trust_remote_code=True,
        max_model_len=4096,
        limit_mm_per_prompt={"image": 1},
        gpu_memory_utilization=0.9,
    )

    sampling_params = SamplingParams(
        temperature=0.2,
        top_p=0.95,
        max_tokens=512,
    )

    conversation = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_pil",
                    "image_pil": image,
                },
                {
                    "type": "text",
                    "text": PROMPT,
                },
            ],
        }
    ]

    outputs = llm.chat(conversation, sampling_params=sampling_params, use_tqdm=False)
    result = outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""

    image_name = Path(IMAGE_PATH).stem
    output_file = Path(OUTPUT_DIR) / f"{image_name}.txt"
    output_file.write_text(result, encoding="utf-8")

    print(result)
    print(f"\n识别结果已保存至: {output_file}")


if __name__ == "__main__":
    main()
