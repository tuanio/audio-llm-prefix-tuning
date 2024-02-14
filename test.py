import torch
from models.panns import (
    Wavegram_Logmel_Cnn14,
    Wavegram_Logmel_Cnn14_Baseline,
)

device = "cuda" if torch.cuda.is_available() else "cpu"

sample_rate = 32000
window_size = 1024
hop_size = 320
mel_bins = 64
fmin = 50
fmax = 14000
classes_num = 527

model = Wavegram_Logmel_Cnn14_Baseline(
    sample_rate=sample_rate,
    window_size=window_size,
    hop_size=hop_size,
    mel_bins=mel_bins,
    fmin=fmin,
    fmax=fmax,
    classes_num=classes_num,
)
load_res = model.load_pretrained(
    "resources/Wavegram_Logmel_Cnn14_mAP=0.439.pth", device
)
print(model)
print(load_res)
print(sum(p.numel() for p in model.parameters()))

x = torch.rand(4, 5 * 32000)

temp_embeds, glob_embeds = model(x)

print(temp_embeds.size(), glob_embeds.size())
