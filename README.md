# Transcription_Whisper

transcribe audio using NVIDIA NeMo Parakeet models.

Input audio is automatically prepared (converted to mono 16 kHz, 16-bit
WAV) using ffmpeg/pydub so Parakeet and Sortformer receive the expected
format.

## Saving a NeMo model locally

You can download and persist a NeMo ASR pretrained model as a .nemo file
using the helper script at `Model/save_model.py`.

```powershell
python -m Model.save_model \
  --model nvidia/parakeet-tdt-0.6b-v2 \
  --out-dir .\models
```
