"""Environment and dependency verification script for SIH26158."""

import sys
import platform
import shutil

def check_environment() -> bool:
    print("=" * 60)
    print("SIH26158 Environment Verification (Phase 0)")
    print("=" * 60)
    print(f"Python Version: {platform.python_version()} ({sys.executable})")
    print(f"OS / Platform:  {platform.system()} {platform.release()} ({platform.machine()})")
    
    # Check Python version
    major, minor = sys.version_info[:2]
    if major < 3 or (major == 3 and minor < 10):
        print(f"[FAIL] Python 3.10+ is required, found {major}.{minor}")
        return False
    print("[PASS] Python version requirement satisfied (>= 3.10)")

    # Check optional GPU availability
    try:
        import torch  # type: ignore
        cuda_available = torch.cuda.is_available()
        device_name = torch.cuda.get_device_name(0) if cuda_available else "N/A"
        print(f"[INFO] PyTorch: {torch.__version__} | CUDA Available: {cuda_available} | Device: {device_name}")
    except ImportError:
        print("[INFO] PyTorch not installed in current environment (optional in Phase 0 scaffolding).")

    # Check ffmpeg binary for video demuxing
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        print(f"[PASS] FFmpeg binary found: {ffmpeg_path}")
    else:
        print("[INFO] FFmpeg binary not found on PATH (will be required for full video ingestion in Phase 1).")

    print("=" * 60)
    print("Verification completed.")
    return True

if __name__ == "__main__":
    success = check_environment()
    sys.exit(0 if success else 1)
