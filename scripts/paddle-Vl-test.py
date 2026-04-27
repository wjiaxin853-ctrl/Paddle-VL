import os
import threading
import time
from pathlib import Path
from statistics import mean

# 放在最前面，跳过网络检查
os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"
# 强制使用 CPU 并禁用 GPU 加载
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from paddleocr import PaddleOCRVL

ROOT_DIR = Path(__file__).resolve().parent.parent
INPUT_IMAGE = ROOT_DIR / "demo/images/取钱+有文字.jpg"
OUTPUT_DIR = ROOT_DIR / "output/Paddle-VL/取钱+有文字"


def get_rss_bytes() -> int:
    statm_path = Path("/proc/self/statm")
    if statm_path.exists():
        pages = int(statm_path.read_text().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE")

    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if os.uname().sysname == "Darwin":
            return rss
        return rss * 1024
    except Exception:
        return 0


def format_bytes(num_bytes: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} TB"


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

    def stop(self) -> float:
        self._stop_event.set()
        self._thread.join()
        return mean(self.samples) if self.samples else float(get_rss_bytes())


def print_metrics(parse_time_seconds: float, start_memory: int, avg_memory: float):
    memory_delta = avg_memory - start_memory

    print("\n===== 性能指标 =====")
    print(f"解析时间: {parse_time_seconds:.2f} 秒")
    print(f"开始内存(RSS): {format_bytes(start_memory)}")
    print(f"运行期平均内存(RSS): {format_bytes(avg_memory)}")
    print(f"平均内存增量: {format_bytes(memory_delta)}")
    print("Token 速度: 当前脚本基于 PaddleOCR-VL 高层 pipeline，无法准确获取生成 token 数，暂不可测")
    print("===================\n")

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
start_memory = get_rss_bytes()
memory_sampler = MemorySampler()
memory_sampler.start()
parse_start = time.perf_counter()
results = list(pipeline.predict(str(INPUT_IMAGE)))
parse_time_seconds = time.perf_counter() - parse_start
avg_memory = memory_sampler.stop()

print_metrics(parse_time_seconds, start_memory, avg_memory)

for res in results:
    res.print() ## 打印预测的结构化输出
    res.save_to_json(save_path=str(OUTPUT_DIR)) ## 保存当前图像的结构化json结果
    res.save_to_markdown(save_path=str(OUTPUT_DIR)) ## 保存当前图像的markdown格式的结果
