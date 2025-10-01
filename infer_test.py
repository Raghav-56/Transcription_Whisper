# install once: uv pip install -U static-ffmpeg pydub
import static_ffmpeg
from static_ffmpeg import run


# Option A: add to PATH for the current process
static_ffmpeg.add_paths()  # downloads ffmpeg+ffprobe on first call [web:73][web:74]

from pydub import AudioSegment



# Option B: get explicit paths and set pydub
#ffmpeg_path, ffprobe_path = run.get_or_fetch_platform_executables_else_raise()  # returns both [web:73][web:74]
#AudioSegment.converter = ffmpeg_path  # tell pydub where ffmpeg is [web:26]
#AudioSegment.ffprobe = ffprobe_path   # tell pydub where ffprobe is [web:63]

# now run existing code that uses pydub

from nemo.collections.asr.models import ASRModel

model = ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2")



wav = "./WhatsApp Audio 2025-10-01 at 15.43.49.wav"  # 16 kHz mono


res = model.transcribe([wav], batch_size=1, return_hypotheses=True, timestamps=True)



print(res[0].text)
print(res[0].timestamp.keys())  # e.g., 'word', 'segment', 'char'
# print(f"\n {res}")
