from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

# 默认：将模型加载到可用设备上
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "prithivMLmods/Qwen2-VL-OCR-2B-Instruct", torch_dtype="auto", device_map="auto"
)

# 建议启用 flash_attention_2，以获得更好的加速效果和更低的显存占用，
# 尤其适用于多图像和视频场景。
# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     "prithivMLmods/Qwen2-VL-OCR-2B-Instruct",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# 默认处理器
processor = AutoProcessor.from_pretrained("prithivMLmods/Qwen2-VL-OCR-2B-Instruct")

# 模型默认每张图像的视觉 token 数范围是 4-16384。
# 你可以根据需要设置 min_pixels 和 max_pixels，
# 例如将 token 数范围设为 256-1280，以平衡速度和内存占用。
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "请描述这张图片。"},
        ],
    }
]

# 推理前处理
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
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

# ========== 流式输出部分 ==========
from transformers import TextIteratorStreamer
from threading import Thread

# 创建流式输出器
streamer = TextIteratorStreamer(processor, skip_prompt=True, skip_special_tokens=True)

# 准备生成参数
generation_kwargs = dict(
    inputs=inputs,
    max_new_tokens=128,
    streamer=streamer,
    do_sample=False,
    temperature=1.0,
)

# 在独立线程中运行生成
thread = Thread(target=model.generate, kwargs=generation_kwargs)
thread.start()

# 实时输出流式结果
print("流式输出结果：")
for new_text in streamer:
    print(new_text, end="", flush=True)
print()  # 最后换行

# ========== 原有的非流式输出方式（保留作为对比）==========
# generated_ids = model.generate(**inputs, max_new_tokens=128)
# generated_ids_trimmed = [
#     out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
# ]
# output_text = processor.batch_decode(
#     generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
# )
# print(output_text)