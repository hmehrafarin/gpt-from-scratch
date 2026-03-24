import torch

def get_device() -> torch.device:
    """
    Returns the best available device in priority order:
      - CUDA  (NVIDIA GPU — Linux/Windows)
      - MPS   (Apple Silicon — Mac M1/M2/M3)
      - CPU   (fallback — all platforms)
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    
    if torch.backends.mps.is_available():
        return torch.device("mps")
    
    return torch.device("cpu")

