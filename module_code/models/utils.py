from subprocess import check_output
import os
import torch


def seed_everything(seed: int):
    """Sets seeds and also makes cuda deterministic for pytorch."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def has_gpu():
    """Check if this machine has a GPU."""
    # Ref: https://stackoverflow.com/a/67504607/1888794
    try:
        check_output("nvidia-smi")
        return True
    except Exception:
        return False
