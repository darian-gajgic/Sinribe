#!/usr/bin/env python3
"""Independent reference transcription: stock faster-whisper large-v3, single pass, GPU."""
import json, sys, time
from faster_whisper import WhisperModel

if len(sys.argv) < 2:
    sys.exit("usage: transcribe.py <audio-file> [output-prefix]")
AUDIO = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else "claude_transcript"

t0 = time.time()
model = WhisperModel("large-v3", device="cuda", compute_type="float16")
print(f"model loaded in {time.time()-t0:.1f}s", flush=True)

segments, info = model.transcribe(
    AUDIO,
    language="de",
    beam_size=5,
    vad_filter=True,
    vad_parameters=dict(min_silence_duration_ms=500),
    word_timestamps=False,
    condition_on_previous_text=False,   # avoids repetition/hallucination drift on long audio
)
print(f"lang={info.language} p={info.language_probability:.2f} dur={info.duration:.1f}s", flush=True)

segs = []
t1 = time.time()
with open(OUT + ".txt", "w") as ftxt:
    for s in segments:
        segs.append({"start": s.start, "end": s.end, "text": s.text.strip()})
        ftxt.write(f"[{int(s.start)//3600:02d}:{int(s.start)//60%60:02d}:{int(s.start)%60:02d}] {s.text.strip()}\n")
        ftxt.flush()
        if len(segs) % 50 == 0:
            print(f"  {len(segs)} segs, audio t={s.end:.0f}s, elapsed {time.time()-t1:.0f}s", flush=True)

json.dump(segs, open(OUT + ".json", "w"), ensure_ascii=False, indent=1)
print(f"DONE {len(segs)} segments in {time.time()-t1:.1f}s (RTF {(time.time()-t1)/info.duration:.3f})", flush=True)
