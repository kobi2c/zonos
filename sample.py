import torch
import torchaudio
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict
from zonos.utils import DEFAULT_DEVICE as device

# model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-hybrid", device=device)
model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=device)

import soundfile as sf
wav_np, sampling_rate = sf.read("assets/exampleaudio.mp3")
wav = torch.tensor(wav_np).float()
if wav.dim() == 1:
    wav = wav.unsqueeze(0)
else:
    wav = wav.t()

speaker = model.make_speaker_embedding(wav.to(device), sampling_rate)

torch.manual_seed(421)

with open("sample.txt", "r", encoding="utf-8") as f:
    text = f.read().strip()
cond_dict = make_cond_dict(text=text, speaker=speaker, language="en-us")
conditioning = model.prepare_conditioning(cond_dict)

codes = model.generate(conditioning)

wavs = model.autoencoder.decode(codes).cpu()
sf.write("sample.wav", wavs[0].cpu().numpy().T, model.autoencoder.sampling_rate)
