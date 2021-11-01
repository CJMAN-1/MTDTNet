import torch

def is_custom_kernel_supported():
    version_str = str(torch.version.cuda).split(".")
    major = version_str[0]
    minor = version_str[1]
    return int(major) >= 10 and int(minor) >= 1
