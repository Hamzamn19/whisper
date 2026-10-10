# 🎤 Smart Whisper - Intelligent Voice Dictation System

Offline speech-to-text dictation powered by **faster-whisper** (CTranslate2) with CUDA acceleration.
Press a hotkey, speak in any language, and the text is typed automatically.

## 📊 Benchmarks (33.6s audio sample, telephony 8kHz)

### RTX 3060 12GB (current, measured Oct 2026)

| Config | Load | Transcribe | RTF | Speed |
|--------|------|-----------|-----|-------|
| turbo / int8 (legacy) | 3.0s | 1.71s | 0.051 | ~20x realtime |
| turbo / float16 | 1.2s | 1.23s | **0.037** | ~27x realtime ⚡ |
| large-v3 / float16 (default) | cached | 2.39s | 0.071 | ~14x realtime 🌟 |

All configs transcribed the sample with 100% correct output.

### NVIDIA MX330 2GB (old machine, Jan 2026)

| Config | Transcribe (28s audio) | RTF |
|--------|----------------------|-----|
| turbo / int8 (cuda) | ~3s | 0.11x (~9x realtime) |

Even the heaviest model (`large-v3`) on the RTX 3060 is faster than `turbo` was on the MX330.

## 🚀 Usage

1. Press **F8** → start recording 🟢
2. Speak in any language (90+ with auto-detect)
3. Press **F8** again → stop 🔴
4. Wait 1–2 seconds → text is typed automatically! ⌨️

## ⚙️ Configuration

In `smart_whisper_google_optimized.py`:

```python
MODEL_SIZE = "large-v3"  # most accurate model
DEVICE = "cuda"          # GPU acceleration
COMPUTE_TYPE = "float16" # best quality/speed balance on 12GB VRAM
SAMPLE_RATE = 16000
```

Run inside the bundled virtualenv (includes CUDA libraries + `faster-whisper` 1.2.1):

```bash
cd ~/Documents/whisper
.venv/bin/python smart_whisper_google_optimized.py
```

## 📂 Files

- `smart_whisper_google_optimized.py` - main dictation script (F8 hotkey)
- `video_translator.py` / `video_translator_gui.py` - offline video subtitling & translation
- `journal.py`, `glossary.py` - helpers
- `FINAL_SUMMARY.md`, `GOOGLE_KV_COMPRESSION.md` - design notes (Arabic)

## 🛠️ Stack

- **faster-whisper** (CTranslate2) + **CUDA** + `float16`
- **sounddevice** - audio capture, **ydotool** - typing on Wayland
- **systemd** user service + GNOME keyboard shortcut (F8)

## 📈 Resource usage (RTX 3060, large-v3/float16)

- GPU VRAM: ~3 GB of 12 GB
- Speed: 14x faster than realtime — dictation feels instant
