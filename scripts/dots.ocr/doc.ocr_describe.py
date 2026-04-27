import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from qwen_vl_utils import process_vision_info
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model_path = "/workspace/models/dots_ocr"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

image_path = "/workspace/demo/images/手持身份证.jpg"

# 图片描述提示词：
# - 描述图片主要内容
# - 提取可见文字
# - 不捏造不存在的信息

prompt = """Ignore OCR layout parsing and do not output JSON, bbox, category, or structured fields.

Please directly describe the visual content of this image in Chinese.

Requirements:
1. Describe what is shown in the image.
2. Mention the main objects, people, and scene.
3. If there is visible text, briefly mention that text after the description.
4. Output plain Chinese text only.
"""

messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path
                },
                {"type": "text", "text": prompt}
            ]
        }
    ]

# Preparation for inference
text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)

inputs = inputs.to("cuda")
inputs.pop("mm_token_type_ids", None)

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=4096) ## 配置图片描述模型输出上限是4096的tocken
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

output_dir = "/workspace/output/dots_ocr"
os.makedirs(output_dir, exist_ok=True)



result = output_text[0] if isinstance(output_text, list) else str(output_text)

image_name = os.path.splitext(os.path.basename(image_path))[0]

print(result)

output_file = f"{output_dir}/{image_name}_describe.txt"

with open(output_file, "w", encoding="utf-8") as f:
    f.write(result)

print(f"\n描述结果已保存至: {output_file}")