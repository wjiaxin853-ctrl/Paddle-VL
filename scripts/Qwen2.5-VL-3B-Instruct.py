#!/usr/bin/env python3

'''
运行命令：
python Qwen2.5-VL-3B-Instruct.py --image demo/images/双人手持身份证.jpg --device mps --prompt "请描述这张图片的内容，并尽量提取其中可见的文字。"  --dtype float32
'''
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _common import (
    apply_chat_template,
    choose_device,
    choose_dtype,
    ensure_model_dir,
    load_chat_template,
    model_dir_for,
    prepare_output_dir,
    resolve_image_path,
)

MODEL_NAME = "Qwen2.5-VL-3B-Instruct"
DEFAULT_PROMPT = "请描述这张图片的内容，并尽量提取其中可见的文字。"
DEFAULT_TEST_IMAGES = [
    Path("demo/images/手持身份证.jpg"),
    Path("demo/images/双人手持身份证.jpg"),
]


def parse_args():
    parser = argparse.ArgumentParser(description=f"{MODEL_NAME} 本地测试脚本")
    parser.add_argument("--model-dir", default=str(model_dir_for(MODEL_NAME)), help="模型目录")
    parser.add_argument("--image", default=None, help="测试图片路径")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="提示词")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto", help="推理设备")
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "float32", "bfloat16"],
        default="auto",
        help="模型精度；Mac 上建议优先测试 mps+float32",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128, help="最大生成长度")
    parser.add_argument("--do-sample", action="store_true", help="开启采样")
    parser.add_argument("--temperature", type=float, default=0.2, help="采样温度")
    parser.add_argument("--output-file", default=None, help="输出保存路径")
    return parser.parse_args()


def main():
    args = parse_args()
    model_dir = Path(args.model_dir).expanduser().resolve()
    image_path = resolve_image_path(args.image, extra_candidates=DEFAULT_TEST_IMAGES)
    ensure_model_dir(model_dir)

    try:
        import torch
        from PIL import Image
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    except ImportError as exc:
        raise RuntimeError("缺少依赖，请先安装 torch、transformers、torchvision、Pillow、accelerate。") from exc

    device = choose_device(args.device, torch)
    if args.dtype == "auto":
        if device == "mps":
            # MPS + float16 在部分机器上会出现重复字符或乱码，测试时优先切到 float32。
            dtype = torch.float32
        else:
            dtype = choose_dtype(device, torch)
    else:
        dtype = getattr(torch, args.dtype)

    print(f"使用设备: {device}")
    print(f"模型精度: {dtype}")

    processor = AutoProcessor.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        use_fast=False,
    )
    chat_template = load_chat_template(model_dir)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(model_dir),
        dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(device).eval()

    image = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": args.prompt},
            ],
        }
    ]
    text = apply_chat_template(processor.tokenizer, messages, chat_template)

    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    ).to(device)

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
    }
    if args.do_sample:
        generation_kwargs["temperature"] = args.temperature

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, **generation_kwargs)

    prompt_length = inputs.input_ids.shape[1]
    generated_ids = [output_ids[prompt_length:] for output_ids in generated_ids]
    output_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    print(f"图片: {image_path}")
    print("\n===== 模型输出 =====\n")
    print(output_text)

    if args.output_file is None:
        output_dir = prepare_output_dir(MODEL_NAME)
        output_file = output_dir / f"{image_path.stem}.txt"
    else:
        output_file = Path(args.output_file).expanduser()
        if not output_file.is_absolute():
            output_file = (Path(__file__).resolve().parent.parent / output_file).resolve()
        output_file.parent.mkdir(parents=True, exist_ok=True)

    output_file.write_text(output_text + "\n", encoding="utf-8")
    print(f"\n输出已保存到: {output_file}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"运行失败: {exc}", file=sys.stderr)
        sys.exit(1)
