# 🎤 smart-whisper

[![CI](https://github.com/Hamzamn19/whisper/actions/workflows/ci.yml/badge.svg)](https://github.com/Hamzamn19/whisper/actions/workflows/ci.yml)

**Local, offline voice dictation and video subtitling with `faster-whisper` on NVIDIA GPUs.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-10%20passed-brightgreen)](#-testing)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Offline](https://img.shields.io/badge/100%25-offline-orange)](#-privacy--offline)

> Your voice never leaves your machine. No cloud APIs, no telemetry — everything runs on your own GPU through local [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2).

**Author:** hamzah sheikh alashrah

---

## ✨ Highlights

- **Press F8, speak, done** — record with a hotkey, talk in any of 90+ languages (auto-detected), and the text is typed automatically where your cursor is.
- **Faster than realtime, even with the biggest model** — measured **RTF 0.037–0.071 on an RTX 3060** (14–27x realtime), so dictation feels instant.
- **Best quality by default** — `large-v3` in `float16`: the most accurate Whisper model, at ~3 GB VRAM of 12 GB.
- **Video subtitling & translation** — `video_translator.py` transcribes and translates whole videos to SRT, with glossary-aware technical terms and a run journal.
- **Offline-first** — models download once from HuggingFace, then run forever without internet.

## 🏗 Architecture

```
[ 🎙 Mic (F8) ] ──> [ faster-whisper large-v3/float16 + CUDA ] ──> [ ydotool typing ]
                              │
                     [ Video files ] ──> [ transcribe ] ──> [ glossary sanitize ] ──> [ SRT + journal ]
```

## 🚀 Quick start

### 1. Prerequisites

- Linux + NVIDIA GPU with ~4 GB free VRAM (tested on RTX 3060 12 GB and MX330 2 GB).
- Python **3.9+**, `ydotool` running (for auto-typing on Wayland).

### 2. Install

```bash
git clone https://github.com/Hamzamn19/whisper.git
cd whisper
uv venv .venv && uv pip install faster-whisper sounddevice
# CUDA users: also install torch so CTranslate2 finds libcublas:
uv pip install torch
```

### 3. Dictate

```bash
.venv/bin/python smart_whisper_google_optimized.py
# press F8 to record, F8 again to stop → text is typed automatically
```

### 4. Subtitle a video

```bash
.venv/bin/python video_translator.py input.mp4 --lang ar
```

| Option | Meaning |
|--------|---------|
| `--lang` | target subtitle language (default: auto-detect source) |
| `MODEL_SIZE` / `COMPUTE_TYPE` | model and precision in the script header (default `large-v3` / `float16`) |

## 📊 Benchmarks (33.6 s telephony audio, 8 kHz)

### RTX 3060 12 GB (current, measured Oct 2026)

| Config | Transcribe | RTF | Speed |
|--------|-----------|-----|-------|
| turbo / int8 (legacy) | 1.71 s | 0.051 | ~20x realtime |
| turbo / float16 | 1.23 s | **0.037** | ~27x realtime ⚡ |
| large-v3 / float16 (default) | 2.39 s | 0.071 | ~14x realtime 🌟 |

All configs transcribed the sample with correct output.

### NVIDIA MX330 2 GB (old machine, Jan 2026)

| Config | Transcribe (28 s audio) | RTF |
|--------|------------------------|-----|
| turbo / int8 (cuda) | ~3 s | 0.11x (~9x realtime) |

> Even the heaviest, most accurate model (`large-v3`) on the RTX 3060 is faster than `turbo` was on the MX330.

## 🗂 Project layout

```
smart_whisper_google_optimized.py   # main dictation script (F8 hotkey, large-v3/float16)
video_translator.py                 # offline video subtitling & translation pipeline
video_translator_gui.py             # GUI for the translator
glossary.py + glossaries/           # technical-term sanitization profiles (tech, general)
journal.py                          # per-run transcription journal
tests/                              # 10 pytest tests (glossary + config consistency)
```

## 🧪 Testing

```bash
uv pip install pytest
.venv/bin/python -m pytest tests/ -v   # 10 passed, CPU-only, no model download
```

The suite covers the **deterministic** parts only: glossary sanitization and
config consistency (model/precision settings match across scripts). Model
inference is intentionally not tested (needs GPU + downloads).

## 🔐 Privacy & offline

- No telemetry, no cloud calls: only localhost + a one-time HuggingFace model download.
- Your audio, transcripts and journals are ignored by `.gitignore`
  (`*.wav`, `*.mp3`, `output/`, `downloads/`, `*.log`).

## 📄 License

[MIT](LICENSE) © 2026 **hamzah sheikh alashrah**

---
---

# 🎤 smart-whisper (العربية)

**إملاء صوتي وترجمة فيديو محلياً وبدون إنترنت عبر `faster-whisper` على كرت NVIDIA.**

> صوتك لا يغادر جهازك أبداً — لا سحابة، لا اتصال خارجي، كل شيء يعمل على كرتك محلياً.

**المؤلف:** hamzah sheikh alashrah

## ✨ المميزات

- **اضغط F8 وتكلم فقط** — تسجيل بزر اختصار، بأي لغة من 90+ لغة (كشف تلقائي)، والنص يُكتب تلقائياً مكان المؤشر.
- **أسرع من الزمن الحقيقي حتى مع أكبر نموذج** — **RTF 0.037–0.071 على RTX 3060** (14–27x)، فالإملاء لحظي.
- **أعلى جودة افتراضياً** — `large-v3` بدقة `float16`: أدق نموذج Whisper، بحوالي 3GB من 12GB.
- **ترجمة الفيديو** — تفريغ وترجمة الأفلام لملفات SRT مع قاموس مصطلحات تقنية وسجل تشغيل.
- **يعمل بدون إنترنت** — النماذج تُحمَّل مرة واحدة ثم تعمل للأبد.

## 🚀 البدء السريع

```bash
git clone https://github.com/Hamzamn19/whisper.git
cd whisper
uv venv .venv && uv pip install faster-whisper sounddevice torch
.venv/bin/python smart_whisper_google_optimized.py
# اضغط F8 للتسجيل، وF8 مرة ثانية للإيقاف ← النص يُكتب تلقائياً
```

## 📊 نتائج القياس (RTX 3060 — عينة 33.6 ثانية)

| الإعداد | التفريغ | RTF |
|---------|---------|-----|
| turbo / int8 (القديم) | 1.71 ث | 0.051 |
| turbo / float16 | 1.23 ث | **0.037** ⚡ |
| large-v3 / float16 (الافتراضي) | 2.39 ث | 0.071 🌟 |

> حتى أثقل نموذج وأدقه (`large-v3`) على الـ 3060 أسرع من `turbo` على الـ MX330 القديم (0.11).
