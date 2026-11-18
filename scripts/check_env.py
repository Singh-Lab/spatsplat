import os
import torch
import time
from dotenv import load_dotenv, find_dotenv

def check_spatsplat_env():
    print("--- SpatSplat Environment Check ---")
    
    # 1. Check .env Loading
    env_path = find_dotenv()
    if env_path:
        load_dotenv(env_path)
        print(f"[OK] Found .env at: {env_path}")
    else:
        print("[ERROR] Could not find .env file in project root.")
        return

    data_dir = os.getenv("SPATSPLAT_DATA_DIR")
    if data_dir:
        print(f"[OK] Data Directory from .env: {data_dir}")
        if os.path.exists(data_dir):
            print(f"[OK] Data path exists on disk.")
        else:
            print(f"[WARNING] Data path defined in .env but folder does not exist.")
    else:
        print("[ERROR] SPATSPLAT_DATA_DIR not found in .env.")

    print("\n--- Hardware Check ---")

    # 2. Check GPU Accessibility (The 5090 Test)
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        vram_total = torch.cuda.get_device_properties(current_device).total_memory / 1024**3
        
        print(f"[OK] CUDA is available ({device_count} device(s) found).")
        print(f"[DEVICE] Current GPU: {device_name}")
        print(f"[MEMORY] Dedicated VRAM: {vram_total:.2f} GB")
        
        # Simple Tensor Operation Test
        # Pre-warm the GPU
        x = torch.rand(2000, 2000).to('cuda')
        y = torch.rand(2000, 2000).to('cuda')
        
        torch.cuda.synchronize()
        start = time.time()
        z = torch.matmul(x, y)
        torch.cuda.synchronize()
        end = time.time()
        
        print(f"[PERF] Tensor Core Test (2000x2000 matmul): {(end-start)*1000:.4f} ms")
    else:
        print("[ERROR] CUDA is NOT available to PyTorch. Check drivers.")

    print("\n--- Library Check ---")
    
    # 3. Check for specific Bioinformatics tools
    try:
        import scanpy as sc
        import squidpy as sq
        print(f"[OK] Scanpy version: {sc.__version__}")
        print(f"[OK] Squidpy version: {sq.__version__}")
    except ImportError as e:
        print(f"[ERROR] Missing bioinformatics libraries: {e}")

    print("\n--- Done ---")

if __name__ == "__main__":
    check_spatsplat_env()