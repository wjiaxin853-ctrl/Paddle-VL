import os

# 放在最前面，跳过网络检查
os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"
# 强制使用 CPU 并禁用 GPU 加载
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from paddleocr import PaddleOCRVL

# 初始化时关闭版面检测，避免加载 PP-DocLayoutV2 模型（节省内存）
pipeline = PaddleOCRVL(
    use_layout_detection=True, ## 需要开启布局检测，占用内存大约8g，否则效果不好
    device='cpu',
    enable_mkldnn=True,
    cpu_threads=4
)
# pipeline = PaddleOCRVL(use_doc_orientation_classify=True) # 通过 use_doc_orientation_classify 指定是否使用文档方向分类模型
# pipeline = PaddleOCRVL(use_doc_unwarping=True) # 通过 use_doc_unwarping 指定是否使用文本图像矫正模块
# pipeline = PaddleOCRVL(use_layout_detection=False) # 通过 use_layout_detection 指定是否使用版面区域检测排序模块
output = pipeline.predict("./手持身份证.jpg")
for res in output:
    res.print() ## 打印预测的结构化输出
    res.save_to_json(save_path="output") ## 保存当前图像的结构化json结果
    res.save_to_markdown(save_path="output") ## 保存当前图像的markdown格式的结果
