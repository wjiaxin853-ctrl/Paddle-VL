import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from qwen_vl_utils import process_vision_info
import os
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model_path = "/workspace/models/dots_ocr"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

image_path = "/workspace/demo/images/取钱+有文字.jpg"

prompt = """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.

5. Final Output: The entire output must be a single JSON object.
"""
# 版面解析提示词说明：
# 1. 要求模型输出整张文档图像中的版面信息
# 2. 每个版面元素都要包含：
#    - bbox：边界框坐标 [x1, y1, x2, y2]
#    - category：版面类别
#    - text：该区域中的文字内容
# 3. 支持的版面类别包括：
#    Caption（图注）、Footnote（脚注）、Formula（公式）、List-item（列表项）、
#    Page-footer（页脚）、Page-header（页眉）、Picture（图片）、
#    Section-header（章节标题）、Table（表格）、Text（正文）、Title（标题）
# 4. 文本格式要求：
#    - Picture 类别不需要 text 字段
#    - Formula 的 text 要用 LaTeX 格式
#    - Table 的 text 要用 HTML 格式
#    - 其他类别的 text 用 Markdown 格式
# 5. 额外约束：
#    - 输出文字必须是图片原文，不能翻译
#    - 所有版面元素必须按人类阅读顺序排序
# 6. 最终输出必须是单个 JSON 对象

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
generated_ids = model.generate(**inputs, max_new_tokens=24000)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

output_dir = "/workspace/output/dots_ocr"
os.makedirs(output_dir, exist_ok=True)



result = output_text[0] if isinstance(output_text, list) else str(output_text)
data = json.loads(result)
texts = [item["text"] for item in data if "text" in item]
ocr_text = "\n".join(texts)
print(ocr_text)

image_name = os.path.splitext(os.path.basename(image_path))[0]
output_file = f"{output_dir}/{image_name}.json"

with open(output_file, "w", encoding="utf-8") as f:
    f.write(result)

text_output_file = f"{output_dir}/{image_name}.txt"
with open(text_output_file, "w", encoding="utf-8") as f:
    f.write(ocr_text)

print(f"\n识别结果已保存至: {output_file}")