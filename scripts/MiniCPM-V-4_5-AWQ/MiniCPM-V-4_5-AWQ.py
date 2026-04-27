import torch
from PIL import Image
from modelscope import AutoModel, AutoTokenizer
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.manual_seed(100)

MODEL_PATH = "/workspace/models/MiniCPM-V-4_5-AWQ"
model = AutoModel.from_pretrained(MODEL_PATH , trust_remote_code=True, # or openbmb/MiniCPM-o-2_6
    attn_implementation='sdpa', torch_dtype=torch.bfloat16) # sdpa or flash_attention_2, no eager
model = model.eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH , trust_remote_code=True) # or openbmb/MiniCPM-o-2_6

IMAGE_PATH = "/workspace/demo/images/手持身份证.jpg"
image = Image.open(IMAGE_PATH).convert('RGB')

enable_thinking=False # If `enable_thinking=True`, the thinking mode is enabled.
stream=True # If `stream=True`, the answer is string

# First round chat
question = "请识别图片中的全部文字。"
# question = "请描述这张图片的主要内容"
msgs = [{'role': 'user', 'content': [image, question]}]

answer = model.chat(
    msgs=msgs,
    tokenizer=tokenizer,
    enable_thinking=enable_thinking,
    stream=True
)

generated_text = ""
for new_text in answer:
    generated_text += new_text
    print(new_text, flush=True, end='')

output_dir = "/workspace/output/MiniCPM-V-4_5-AWQ"
os.makedirs(output_dir, exist_ok=True)
image_name = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
output_file = f"{output_dir}/{image_name}.txt"

with open(output_file, "w", encoding="utf-8") as f:
    f.write(generated_text)

print(f"\n识别结果已保存至: {output_file}")