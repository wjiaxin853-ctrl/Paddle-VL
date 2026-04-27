from transformers import AutoModel, AutoTokenizer
import torch
import os
import types

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
model_name = "/workspace/models/DeepSeek-OCR-2"
image_file = "/workspace/demo/images/取钱+有文字.jpg"
output_path = "/workspace/output/DeepSeek-OCR-2"

os.makedirs(output_path, exist_ok=True)

if torch.cuda.is_available():
    device = "cuda"
    torch_dtype = torch.bfloat16
    print("✅ 使用 CUDA 加速")
else:
    device = "cpu"
    torch_dtype = torch.float32
    print("⚠️ 使用 CPU 推理")

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True, use_safetensors=True,torch_dtype=torch_dtype,)
model = model.eval().to(device)


def to_device(self, *args, **kwargs):
    return self.to(device)
if hasattr(model, "cuda"):
    model.cuda = types.MethodType(to_device, model)


# prompt = "<image>\nFree OCR. "
prompt = "<image>\n请描述这张图片中可见的主体内容。"

res = model.infer(
    tokenizer,
    prompt=prompt,
    image_file=image_file,
    output_path = output_path,
    base_size = 1024,
    image_size = 768,
    crop_mode=True,
    save_results = True
)

print("\n识别结果已保存至:", output_path)
print(res)