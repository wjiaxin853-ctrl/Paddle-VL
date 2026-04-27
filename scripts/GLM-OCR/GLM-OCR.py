import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from modelscope import AutoProcessor, AutoModelForImageTextToText
import torch

MODEL_PATH = "/workspace/models/GLM-OCR"
IMAGE_PATH = "/workspace/demo/images/手持身份证.jpg"
OUTPUT_DIR = "/workspace/output/GLM-OCR"

os.makedirs(OUTPUT_DIR, exist_ok=True)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "url": IMAGE_PATH,
            },
            {
                "type": "text",
                "text": "Text Recognition:",
            },
        ],
    }
]

processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    pretrained_model_name_or_path=MODEL_PATH,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
)

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

inputs.pop("token_type_ids", None)

generated_ids = model.generate(**inputs, max_new_tokens=8192)
output_text = processor.decode(
    generated_ids[0][inputs["input_ids"].shape[1]:],
    skip_special_tokens=False,
)

image_name = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
output_file = f"{OUTPUT_DIR}/{image_name}.txt"

with open(output_file, "w", encoding="utf-8") as f:
    f.write(output_text)

print(output_text)
print(f"\n识别结果已保存至: {output_file}")
