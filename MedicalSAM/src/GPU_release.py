import gc

import torch


def release_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()
    print("✅ GPU 記憶體已釋放")


if __name__ == "__main__":
    release_gpu_memory()
