import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"

print(device)