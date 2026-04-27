#!/usr/bin/env python3

'''
运行命令：
【测试文件描述】
python Qwen2.5-VL-3B-Instruct.py --image demo/images/双人手持身份证.jpg --device mps --prompt "请描述这张图片的内容，并尽量提取其中可见的文字。"  --dtype float32

【测试文件解析】
--output-file
'''
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from statistics import mean

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts._common import (
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
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def get_rss_bytes() -> int:
    try:
        import psutil

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        pass

    status_path = Path("/proc/self/status")
    if status_path.exists():
        try:
            for line in status_path.read_text().splitlines():
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
        except Exception:
            pass

    statm_path = Path("/proc/self/statm")
    if statm_path.exists():
        try:
            parts = statm_path.read_text().split()
            if len(parts) > 1:
                return int(parts[1]) * os.sysconf("SC_PAGE_SIZE")
        except Exception:
            pass

    try:
        completed = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(os.getpid())],
            capture_output=True,
            text=True,
            check=True,
        )
        rss_kb_text = completed.stdout.strip()
        if rss_kb_text:
            return int(rss_kb_text) * 1024
    except Exception:
        pass

    return 0


def format_bytes(num_bytes: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    sign = "-" if value < 0 else ""
    value = abs(value)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{sign}{value:.2f} {unit}"
        value /= 1024
    return f"{sign}{value:.2f} TB"


class MemorySampler:
    def __init__(self, interval_seconds: float = 0.1):
        self.interval_seconds = interval_seconds
        self.samples: list[int] = []
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop_event.is_set():
            self.samples.append(get_rss_bytes())
            self._stop_event.wait(self.interval_seconds)
        self.samples.append(get_rss_bytes())

    def start(self):
        self.samples = [get_rss_bytes()]
        self._thread.start()

    def stop(self) -> list[int]:
        self._stop_event.set()
        self._thread.join()
        return self.samples if self.samples else [get_rss_bytes()]


def print_metrics(
    total_time_seconds: float,
    generation_time_seconds: float,
    start_memory: int,
    avg_memory: float,
    peak_memory: int,
    generated_token_count: int,
):
    token_speed = generated_token_count / generation_time_seconds if generation_time_seconds > 0 else 0.0

    print("\n===== 性能指标 =====")
    print(f"解析时间(总): {total_time_seconds:.2f} 秒")
    print(f"生成时间: {generation_time_seconds:.2f} 秒")
    print(f"开始内存(RSS): {format_bytes(start_memory)}")
    print(f"运行期平均内存(RSS): {format_bytes(avg_memory)}")
    print(f"占用内存(进程峰值RSS): {format_bytes(peak_memory)}")
    print(f"生成 token 数: {generated_token_count}")
    print(f"Token 输出速度: {token_speed:.2f} token/s")
    print("===================\n")


def resolve_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path


def collect_image_paths(image_texts: list[str] | None, image_dir_text: str | None) -> list[Path]:
    image_paths: list[Path] = []

    for image_text in image_texts or []:
        image_path = resolve_path(image_text)
        if not image_path.exists():
            raise FileNotFoundError(f"图片不存在: {image_path}")
        if image_path.is_dir():
            raise IsADirectoryError(f"--image 需要传图片文件，不是目录: {image_path}")
        image_paths.append(image_path)

    if image_dir_text:
        image_dir = resolve_path(image_dir_text)
        if not image_dir.exists():
            raise FileNotFoundError(f"图片目录不存在: {image_dir}")
        if not image_dir.is_dir():
            raise NotADirectoryError(f"--image-dir 需要传目录路径: {image_dir}")

        dir_image_paths = sorted(
            path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not dir_image_paths:
            raise FileNotFoundError(f"图片目录中没有找到支持的图片文件: {image_dir}")
        image_paths.extend(dir_image_paths)

    if image_paths:
        return image_paths

    return [resolve_image_path(None, extra_candidates=DEFAULT_TEST_IMAGES)]


def default_output_name(image_paths: list[Path]) -> str:
    if len(image_paths) == 1:
        return f"{image_paths[0].stem}.txt"
    return f"{image_paths[0].stem}_to_{image_paths[-1].stem}.txt"


def resolve_output_path(output_text: str | None, image_paths: list[Path]) -> Path:
    if output_text is None:
        return prepare_output_dir(MODEL_NAME) / default_output_name(image_paths)

    output_path = Path(output_text).expanduser()
    if not output_path.is_absolute():
        output_path = (REPO_ROOT / output_path).resolve()

    if output_path.exists() and output_path.is_dir():
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path / default_output_name(image_paths)

    if output_path.suffix.lower() == ".txt":
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path

    output_path.mkdir(parents=True, exist_ok=True)
    return output_path / default_output_name(image_paths)


def parse_args():
    parser = argparse.ArgumentParser(description=f"{MODEL_NAME} 本地测试脚本")
    parser.add_argument("--model-dir", default=str(model_dir_for(MODEL_NAME)), help="模型目录")
    parser.add_argument("--image", action="append", default=None, help="测试图片路径；可多次传入")
    parser.add_argument("--image-dir", default=None, help="批量图片目录")
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
    parser.add_argument("--output-file", default=None, help="输出保存路径；可传 txt 文件或目录")
    return parser.parse_args()


def main():
    args = parse_args()
    model_dir = Path(args.model_dir).expanduser().resolve()
    image_paths = collect_image_paths(args.image, args.image_dir)
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
    print("输入图片:")
    for index, image_path in enumerate(image_paths, start=1):
        print(f"  {index}. {image_path}")

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

    start_memory = get_rss_bytes()
    memory_sampler = MemorySampler()
    memory_sampler.start()
    total_start = time.perf_counter()

    total_generation_time_seconds = 0.0
    total_generated_token_count = 0
    output_sections: list[str] = []

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
    }
    if args.do_sample:
        generation_kwargs["temperature"] = args.temperature

    for index, image_path in enumerate(image_paths, start=1):
        with Image.open(image_path) as opened_image:
            image = opened_image.convert("RGB")

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

        generation_start = time.perf_counter()
        with torch.inference_mode():
            full_generated_ids = model.generate(**inputs, **generation_kwargs)
        generation_time_seconds = time.perf_counter() - generation_start
        total_generation_time_seconds += generation_time_seconds

        prompt_length = inputs.input_ids.shape[1]
        generated_token_count = int(full_generated_ids.shape[-1] - prompt_length)
        total_generated_token_count += generated_token_count
        generated_ids = [output_ids[prompt_length:] for output_ids in full_generated_ids]
        output_text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        print(f"\n===== 模型输出 {index}/{len(image_paths)} =====")
        print(f"图片: {image_path}\n")
        print(output_text)

        output_sections.append(f"【图片{index}】\n路径: {image_path}\n\n{output_text}")

    total_time_seconds = time.perf_counter() - total_start
    memory_samples = memory_sampler.stop()
    avg_memory = mean(memory_samples)
    peak_memory = max(memory_samples)

    output_file = resolve_output_path(args.output_file, image_paths)
    output_text = "\n\n".join(output_sections).strip()
    output_file.write_text(output_text + "\n", encoding="utf-8")
    print(f"\n输出已保存到: {output_file}")
    print_metrics(
        total_time_seconds=total_time_seconds,
        generation_time_seconds=total_generation_time_seconds,
        start_memory=start_memory,
        avg_memory=avg_memory,
        peak_memory=peak_memory,
        generated_token_count=total_generated_token_count,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"运行失败: {exc}", file=sys.stderr)
        sys.exit(1)
