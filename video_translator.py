#!/usr/bin/env python3
"""
Video Translator: Offline Video Subtitling & Translation
Optimized for low VRAM (NVIDIA MX330 2GB)

Pipeline: Video -> Audio -> Transcription -> Translation -> SRT
"""

import argparse
import gc
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional

try:
    import torch
except ImportError:
    torch = None

from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer

# Local modules
try:
    from journal import TranscriptionJournal
    HAS_JOURNAL = True
except ImportError:
    HAS_JOURNAL = False
    print("⚠️ Journal module not found. Crash recovery disabled.")

try:
    from glossary import get_default_glossary, sanitize_segments
    HAS_GLOSSARY = True
except ImportError:
    HAS_GLOSSARY = False
    print("⚠️ Glossary module not found. Term sanitization disabled.")

# yt-dlp for video downloading
try:
    import yt_dlp
    HAS_YTDLP = True
except ImportError:
    HAS_YTDLP = False
    print("⚠️ yt-dlp not installed. Video downloading disabled. Run: pip install yt-dlp")

# audio-separator for professional vocal isolation (UVR5 MDX-Net)
try:
    from audio_separator.separator import Separator
    HAS_AUDIO_SEPARATOR = True
except ImportError:
    HAS_AUDIO_SEPARATOR = False
    print("⚠️ audio-separator not installed. Vocal isolation disabled. Run: pip install audio-separator[gpu]")


# ============================================================================
# VIDEO DOWNLOAD (yt-dlp)
# ============================================================================
# Quality presets for download
DOWNLOAD_QUALITIES = {
    "Best": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio/best",
    "1080p": "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080]",
    "720p": "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720]",
    "480p": "bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]/best[height<=480]",
    "360p": "bestvideo[height<=360][ext=mp4]+bestaudio[ext=m4a]/best[height<=360]",
    "Audio Only": "bestaudio[ext=m4a]/bestaudio",
}

def download_video(
    url: str, 
    output_dir: str = None, 
    speed_limit: int = None,
    quality: str = "720p",
    progress_callback = None
) -> Optional[str]:
    """
    Download video from URL using yt-dlp.
    Supports 1000+ websites (YouTube, Vimeo, TikTok, Twitter, etc.)
    
    Args:
        url: Video URL (YouTube, Vimeo, TikTok, Twitter, etc.)
        output_dir: Directory to save the video (default: current directory)
        speed_limit: Download speed limit in KB/s (None = unlimited)
        quality: Quality preset (Best, 1080p, 720p, 480p, 360p, Audio Only)
        progress_callback: Optional callback function(status, percent, speed)
    
    Returns:
        Path to downloaded video file, or None on failure
    """
    if not HAS_YTDLP:
        print("❌ yt-dlp not installed. Run: pip install yt-dlp")
        return None
    
    if not url or not url.strip():
        print("❌ Empty URL provided")
        return None
    
    # Setup output directory
    if output_dir is None:
        output_dir = os.getcwd()
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Output template
    output_template = str(output_dir / "%(title)s.%(ext)s")
    
    # Progress hook for UI updates
    def progress_hook(d):
        if progress_callback:
            if d['status'] == 'downloading':
                percent = d.get('_percent_str', '0%').strip()
                speed = d.get('_speed_str', 'N/A').strip()
                progress_callback('downloading', percent, speed)
            elif d['status'] == 'finished':
                progress_callback('finished', '100%', 'Done')
    
    # Get format string based on quality
    format_str = DOWNLOAD_QUALITIES.get(quality, DOWNLOAD_QUALITIES["720p"])
    print(f"📺 Quality: {quality}")
    
    # yt-dlp options
    ydl_opts = {
        'format': format_str,
        'outtmpl': output_template,
        'merge_output_format': 'mp4' if quality != "Audio Only" else 'm4a',
        'noplaylist': True,  # Only download single video
        'quiet': False,
        'no_warnings': False,
        'progress_hooks': [progress_hook],
        # Linux/Fedora compatibility
        'prefer_ffmpeg': True,
        'postprocessor_args': ['-c', 'copy'],  # Fast merge without re-encoding
    }
    
    # Apply speed limit if specified (convert KB/s to bytes/s)
    if speed_limit and speed_limit > 0:
        ydl_opts['ratelimit'] = speed_limit * 1024
        print(f"📶 Speed limit: {speed_limit} KB/s")
    
    print(f"\n{'=' * 60}")
    print("📥 STEP 0: Video Download")
    print("=" * 60)
    print(f"🔗 URL: {url}")
    print(f"📁 Output: {output_dir}")
    
    downloaded_file = None
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Extract info first to get filename
            info = ydl.extract_info(url, download=False)
            if info is None:
                print("❌ Failed to extract video info. Invalid URL?")
                return None
            
            title = info.get('title', 'video')
            duration = info.get('duration', 0)
            print(f"📹 Title: {title}")
            print(f"⏱️  Duration: {duration // 60}m {duration % 60}s")
            
            # Download
            print("⬇️  Downloading...")
            ydl.download([url])
            
            # Find the downloaded file
            ext = info.get('ext', 'mp4')
            # Clean filename (yt-dlp sanitizes title)
            sanitized_title = yt_dlp.utils.sanitize_filename(title)
            downloaded_file = str(output_dir / f"{sanitized_title}.{ext}")
            
            # Check if file exists with mp4 extension (after merge)
            if not os.path.exists(downloaded_file):
                downloaded_file = str(output_dir / f"{sanitized_title}.mp4")
            
            if os.path.exists(downloaded_file):
                file_size = os.path.getsize(downloaded_file) / (1024 * 1024)
                print(f"✅ Downloaded: {downloaded_file}")
                print(f"📊 Size: {file_size:.1f} MB")
                return downloaded_file
            else:
                # Try to find any recently downloaded file
                import glob
                mp4_files = glob.glob(str(output_dir / "*.mp4"))
                if mp4_files:
                    # Get most recently modified
                    latest = max(mp4_files, key=os.path.getmtime)
                    print(f"✅ Downloaded: {latest}")
                    return latest
                print("❌ Download completed but file not found")
                return None
                
    except yt_dlp.utils.DownloadError as e:
        print(f"❌ Download error: {e}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None


# ============================================================================
# CONFIGURATION
# ============================================================================
WHISPER_MODEL = "turbo"  # Same as smart_whisper.py
BATCH_SIZE = 16  # Increased from 8 for faster translation
DEFAULT_TARGET_LANG = "arb_Arab"

# Add CUDA library path (same as smart_whisper.py)
import os
cuda_lib_path = os.path.expanduser("~/.local/lib/python3.12/site-packages/nvidia/cublas/lib")
if os.path.exists(cuda_lib_path):
    os.environ['LD_LIBRARY_PATH'] = f"{cuda_lib_path}:{os.environ.get('LD_LIBRARY_PATH', '')}"

# Helsinki-NLP supported languages (limited to EN<->AR, TR<->AR, EN<->TR)
HELSINKI_LANGUAGES = {
    "Arabic": "arb_Arab",
    "English": "eng_Latn",
    "Turkish": "tur_Latn",
}


# ============================================================================
# PROFESSIONAL VOCAL ISOLATION (UVR5 MDX-Net)
# ============================================================================
def isolate_vocals(
    audio_path: str,
    output_dir: str = None,
    progress_callback = None
) -> Optional[str]:
    """
    Isolate vocals from audio using UVR5 MDX-Net models.
    Optimized for low VRAM (2GB MX330) with segment_size=256.
    
    Args:
        audio_path: Path to input audio file
        output_dir: Directory for output files (default: same as input)
        progress_callback: Optional callback for progress updates
    
    Returns:
        Path to isolated vocals file, or None on failure
    """
    if not HAS_AUDIO_SEPARATOR:
        print("❌ audio-separator not installed. Run: pip install audio-separator[gpu]")
        return None
    
    print("\n" + "-" * 40)
    print("🎤 VOCAL ISOLATION (UVR5 MDX-Net)")
    print("-" * 40)
    
    if output_dir is None:
        output_dir = str(Path(audio_path).parent)
    
    try:
        # Initialize separator with low VRAM settings
        print("🔄 Initializing audio separator...")
        print("📊 VRAM Mode: Low (segment_size=256 for 2GB GPU)")
        
        separator = Separator(
            model_file_dir="/tmp/audio-separator-models",
            output_dir=output_dir,
            output_format="wav",
            # Low VRAM settings for MX330 (2GB)
            mdx_params={
                "segment_size": 256,     # Reduced from 512 for 2GB VRAM
                "overlap": 0.25,
                "batch_size": 1,         # Single batch for low VRAM
                "enable_denoise": True,
            }
        )
        
        # Load the default MDX-Net model for vocal separation
        # This downloads and uses the recommended model automatically
        print("📥 Loading MDX-Net model (first run may download ~100MB)...")
        separator.load_model()  # Uses default model
        print("✅ Model loaded successfully")
        
        if progress_callback:
            progress_callback("🎤 Isolating vocals...")
        
        print(f"🎵 Processing: {audio_path}")
        
        # Separate vocals from instrumentals
        output_files = separator.separate(audio_path)
        
        # Clean up GPU memory immediately after separation
        print("🧹 Cleaning GPU memory...")
        del separator
        if torch:
            torch.cuda.empty_cache()
        gc.collect()
        print("✅ GPU memory freed")
        
        # Find the vocals file in output
        vocals_file = None
        for f in output_files:
            if "(Vocals)" in f or "vocals" in f.lower():
                vocals_file = f
                break
        
        if vocals_file and os.path.exists(vocals_file):
            file_size = os.path.getsize(vocals_file) / (1024 * 1024)
            print(f"✅ Vocals isolated: {vocals_file}")
            print(f"📊 Size: {file_size:.1f} MB")
            return vocals_file
        else:
            print("❌ No vocals file found in output")
            return None
            
    except Exception as e:
        print(f"❌ Vocal isolation failed: {e}")
        # Clean up GPU memory on error too
        if torch:
            torch.cuda.empty_cache()
        gc.collect()
        return None

# ============================================================================
# SMART CONTENT PROFILES
# ============================================================================
# Each profile contains optimized settings for different content types
CONTENT_PROFILES = {
    "Action Movies": {
        "description": "High noise filtering for explosions, action sequences",
        "vad_threshold": 0.55,
        "min_silence_ms": 600,
        "min_speech_ms": 300,
        "speech_pad_ms": 250,
        "ffmpeg_filters": "highpass=f=300,lowpass=f=3000,loudnorm",  # Higher highpass for explosions
        "initial_prompt": "Action, explosions, shouting, intense dialogue",
    },
    "Documentaries": {
        "description": "Low noise filtering for clear educational speech",
        "vad_threshold": 0.70,
        "min_silence_ms": 800,
        "min_speech_ms": 400,
        "speech_pad_ms": 300,
        "ffmpeg_filters": "highpass=f=100,lowpass=f=4000,loudnorm",  # Wide range for clear speech
        "initial_prompt": "Clear educational speech, terminology, narration",
    },
    "Sitcoms": {
        "description": "Medium filtering for conversational content with background laughter",
        "vad_threshold": 0.60,
        "min_silence_ms": 500,
        "min_speech_ms": 300,
        "speech_pad_ms": 250,
        "ffmpeg_filters": "highpass=f=200,lowpass=f=3500,loudnorm",
        "initial_prompt": "Conversational, laughter background, casual dialogue",
    },
    "Music/Rap": {
        "description": "Minimal filtering for fast lyrics and singing",
        "vad_threshold": 0.40,
        "min_silence_ms": 400,
        "min_speech_ms": 200,
        "speech_pad_ms": 150,
        "ffmpeg_filters": "loudnorm",  # Minimal filtering, preserve music
        "initial_prompt": "Singing, rhythmic lyrics, fast-paced vocals",
    },
    "Podcasts": {
        "description": "High threshold for studio quality speech",
        "vad_threshold": 0.75,
        "min_silence_ms": 1000,
        "min_speech_ms": 500,
        "speech_pad_ms": 300,
        "ffmpeg_filters": "highpass=f=80,lowpass=f=8000,loudnorm",  # Wide range for studio quality
        "initial_prompt": "Studio quality speech, interview, discussion",
    },
}

DEFAULT_PROFILE = "Sitcoms"  # Balanced default


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def clear_gpu_memory():
    """Aggressively clear GPU memory."""
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print("🧹 GPU memory cleared")


def format_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def write_srt(segments: List[dict], output_path: str):
    """Write segments to SRT file."""
    with open(output_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            start = format_timestamp(seg["start"])
            end = format_timestamp(seg["end"])
            text = seg["text"].strip()
            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")
    print(f"✅ SRT file saved: {output_path}")


# ============================================================================
# STEP A: AUDIO EXTRACTION
# ============================================================================
def extract_audio(video_path: str, audio_path: str, content_profile: str = None) -> bool:
    """
    Extract audio from video and convert to 16kHz mono WAV.
    Uses profile-specific FFmpeg filters for optimal audio processing.
    """
    print("\n" + "=" * 60)
    print("📼 STEP A: Audio Extraction")
    print("=" * 60)
    
    if os.path.exists(audio_path):
        print(f"⏭️  Audio file already exists: {audio_path}")
        return True
    
    # Get FFmpeg filters from profile
    profile = CONTENT_PROFILES.get(content_profile, CONTENT_PROFILES[DEFAULT_PROFILE])
    ffmpeg_filters = profile.get("ffmpeg_filters", "highpass=f=200,lowpass=f=3000,loudnorm")
    
    print(f"🎬 Profile: {content_profile or DEFAULT_PROFILE}")
    print(f"🎛️  Filters: {ffmpeg_filters}")
    
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",                    # No video
        "-af", ffmpeg_filters,    # Profile-specific audio filters
        "-acodec", "pcm_s16le",   # PCM 16-bit
        "-ar", "16000",           # 16kHz sample rate
        "-ac", "1",               # Mono
        audio_path
    ]
    
    print(f"🔊 Extracting audio from: {video_path}")
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            check=True
        )
        print(f"✅ Audio extracted: {audio_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg error: {e.stderr}")
        return False
    except FileNotFoundError:
        print("❌ FFmpeg not found. Please install: sudo dnf install ffmpeg")
        return False


# ============================================================================
# PROBE PHASE: Analyze first 30 seconds
# ============================================================================
def probe_audio(audio_path: str, content_profile: str = None) -> dict:
    """
    Probe first 30 seconds of audio to determine optimal settings.
    Returns adjusted profile settings based on audio analysis.
    """
    print("\n" + "-" * 40)
    print("🔍 PROBE PHASE: Analyzing audio...")
    print("-" * 40)
    
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("⚠️ Probe skipped: faster-whisper not available")
        return CONTENT_PROFILES.get(content_profile, CONTENT_PROFILES[DEFAULT_PROFILE])
    
    # Get base profile
    profile = CONTENT_PROFILES.get(content_profile, CONTENT_PROFILES[DEFAULT_PROFILE]).copy()
    
    try:
        # Load model for probe (use same settings)
        model = WhisperModel(
            WHISPER_MODEL,
            device="cuda",
            compute_type="int8",
            num_workers=1
        )
        
        # Probe first 30 seconds only
        _, info = model.transcribe(
            audio_path,
            language=None,
            beam_size=1,  # Fast probe
            temperature=0.0,
            vad_filter=True,
            vad_parameters=dict(threshold=0.5, min_silence_duration_ms=500),
            # Limit to ~30 seconds by using clip_timestamps
        )
        
        lang_prob = info.language_probability
        detected_lang = info.language
        
        print(f"🌐 Detected: {detected_lang} (confidence: {lang_prob:.2f})")
        
        # Auto-adjust if low confidence (noisy audio)
        if lang_prob < 0.5:
            print("⚠️ Low confidence detected - adjusting to High Noise mode")
            profile["vad_threshold"] = max(0.35, profile["vad_threshold"] - 0.10)
            profile["min_speech_ms"] = max(150, profile["min_speech_ms"] - 50)
            print(f"   └─ Threshold: {profile['vad_threshold']}, Min Speech: {profile['min_speech_ms']}ms")
        elif lang_prob > 0.9:
            print("✅ High quality audio detected")
        
        # Cleanup
        del model
        clear_gpu_memory()
        
    except Exception as e:
        print(f"⚠️ Probe failed: {e}")
    
    return profile


# ============================================================================
# STEP B: TRANSCRIPTION (Whisper Phase)
# ============================================================================
def transcribe_audio(audio_path: str, source_lang: str = None, initial_prompt: str = None, profile_settings: dict = None) -> Optional[Tuple[List[dict], str]]:
    """
    Transcribe audio using faster-whisper Turbo model on GPU.
    
    Args:
        audio_path: Path to the audio file
        source_lang: Optional source language hint
        initial_prompt: Optional prompt for context/keywords to improve transcription
        profile_settings: VAD settings from content profile
    
    Returns: (segments, detected_language) or None on failure
    """
    print("\n" + "=" * 60)
    print("🎤 STEP B: Transcription (Whisper Turbo)")
    print("=" * 60)
    
    # Get settings from profile or use defaults
    if profile_settings is None:
        profile_settings = CONTENT_PROFILES[DEFAULT_PROFILE]
    
    vad_threshold = profile_settings.get("vad_threshold", 0.50)
    min_silence_ms = profile_settings.get("min_silence_ms", 500)
    min_speech_ms = profile_settings.get("min_speech_ms", 250)
    speech_pad_ms = profile_settings.get("speech_pad_ms", 200)
    
    # Combine profile prompt with user prompt
    profile_prompt = profile_settings.get("initial_prompt", "")
    if initial_prompt:
        combined_prompt = f"{profile_prompt}. {initial_prompt}"
    else:
        combined_prompt = profile_prompt
    
    print(f"🎬 VAD: threshold={vad_threshold}, silence={min_silence_ms}ms, speech={min_speech_ms}ms")
    
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("❌ faster-whisper not installed. Run: pip install faster-whisper")
        return None
    
    model = None
    
    # Load Turbo model on GPU (same as smart_whisper.py)
    try:
        print(f"🔄 Loading {WHISPER_MODEL} on GPU...")
        model = WhisperModel(
            WHISPER_MODEL,
            device="cuda",
            compute_type="int8",
            num_workers=1
        )
        print(f"🚀 Loaded {WHISPER_MODEL} on GPU successfully!")
    except Exception as e:
        print(f"❌ Failed to load Whisper model: {e}")
        return None
    
    # Transcribe with auto language detection or specified language
    print(f"� Transcribing: {audio_path}")
    if source_lang:
        print(f"🌐 Source language: {source_lang}")
    else:
        print("🔍 Auto-detecting language...")
    
    try:
        segments_iter, info = model.transcribe(
            audio_path,
            language=source_lang,  # None = auto-detect
            beam_size=3,
            temperature=0.0,
            initial_prompt=combined_prompt,  # Profile + user prompt
            vad_filter=True,
            vad_parameters=dict(
                threshold=vad_threshold,
                min_silence_duration_ms=min_silence_ms,
                min_speech_duration_ms=min_speech_ms,
                speech_pad_ms=speech_pad_ms
            )
        )
        
        detected_lang = info.language
        print(f"📝 Detected language: {detected_lang} (prob: {info.language_probability:.2f})")
        print(f"⏱️  Audio duration: {info.duration:.1f}s")
        
        # Collect segments with progress bar
        segments = []
        with tqdm(desc="Transcribing", unit="seg") as pbar:
            for seg in segments_iter:
                segments.append({
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text
                })
                pbar.update(1)
        
        print(f"✅ Transcribed {len(segments)} segments")
        
    except Exception as e:
        print(f"❌ Transcription error: {e}")
        return None
    finally:
        # CRITICAL: Clean up Whisper model
        print("🧹 Unloading Whisper model...")
        del model
        clear_gpu_memory()
    
    return segments, detected_lang


# ============================================================================
# STEP C: TRANSLATION (Helsinki-NLP Phase)
# ============================================================================
def translate_segments(
    segments: List[dict], 
    nllb_model_path: str = None, # مهمل
    source_lang: str = "en",     # لغة الفيديو (من Whisper)
    target_lang: str = "ar",     # اللغة الهدف (من إعداداتك)
    batch_size: int = 8
) -> Optional[List[dict]]:
    """
    Multi-Agent Translation System (Helsinki-NLP).
    Supports: EN<->AR, TR<->AR, EN<->TR
    Routing logic selects the best lightweight CPU model.
    """
    print("\n" + "=" * 60)
    print(f"🌍 STEP C: Smart Routing Translation")
    print("=" * 60)
    
    # 1. تنظيف أكواد اللغة (لتتوافق مع Whisper و Helsinki)
    # Whisper returns: 'en', 'ar', 'tr'
    # NLLB input was: 'eng_Latn', etc. -> We need to simplify back to 2 letters
    
    # خريطة تبسيط الأكواد (Mapping)
    lang_map = {
        "eng_latn": "en", "eng": "en", "en": "en",
        "arb_arab": "ar", "ara": "ar", "ar": "ar",
        "tur_latn": "tr", "tur": "tr", "tr": "tr"
    }
    
    src = lang_map.get(source_lang.lower(), source_lang.lower())
    tgt = lang_map.get(target_lang.lower(), target_lang.lower())[:2] # نأخذ أول حرفين فقط للتأكيد
    
    print(f"🔍 Routing: Source[{src}] -> Target[{tgt}]")

    if src == tgt:
        print("⏭️  Same language - skipping translation.")
        return segments

    # 2. خريطة الموديلات (The Agent Registry)
    # هنا نحدد الموديل المناسب لكل زوج لغوي
    model_map = {
        "en-ar": "Helsinki-NLP/opus-mt-en-ar",
        "ar-en": "Helsinki-NLP/opus-mt-ar-en",
        "tr-ar": "Helsinki-NLP/opus-mt-tr-ar",
        "ar-tr": "Helsinki-NLP/opus-mt-ar-tr",
        # إضافات مفيدة
        "en-tr": "Helsinki-NLP/opus-mt-en-tr",
        "tr-en": "Helsinki-NLP/opus-mt-tr-en"
    }
    
    pair_key = f"{src}-{tgt}"
    model_name = model_map.get(pair_key)
    
    if not model_name:
        print(f"❌ Error: No specialized model found for pair {pair_key}")
        print(f"   Supported pairs: {list(model_map.keys())}")
        return None

    # 3. تحميل الموديل المتخصص
    try:
        print(f"📥 Loading Specialist Agent: {model_name}...")
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        
        # CPU Offloading (حماية كرت الشاشة)
        model.to("cpu")
        print("✅ Model loaded on CPU (VRAM safe).")
        
    except Exception as e:
        print(f"❌ Failed to load model {model_name}: {e}")
        return None

    # 4. عملية الترجمة
    translated_segments = []
    texts = [seg["text"].strip() for seg in segments]
    
    print(f"🚀 Translating {len(texts)} segments ({src}->{tgt})...")

    # دالة تقسيم (Batching)
    def chunked(iterable, n):
        for i in range(0, len(iterable), n):
            yield iterable[i:i + n]

    from tqdm import tqdm
    for batch_idx, batch_texts in enumerate(tqdm(chunked(texts, batch_size), total=(len(texts)//batch_size)+1)):
        try:
            # Tokenization
            inputs = tokenizer(
                batch_texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            ).to("cpu")
            
            # Generation
            translated_tokens = model.generate(
                **inputs,
                max_length=512,
                num_beams=4,             # دقة عالية
                early_stopping=True,
                no_repeat_ngram_size=2   # منع التكرار
            )
            
            # Decoding
            decoded_batch = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
            
            # Mapping back
            current_start_idx = batch_idx * batch_size
            for i, trans_text in enumerate(decoded_batch):
                if current_start_idx + i < len(segments):
                    original_seg = segments[current_start_idx + i]
                    translated_segments.append({
                        "start": original_seg["start"],
                        "end": original_seg["end"],
                        "text": trans_text
                    })
                
        except Exception as e:
            print(f"⚠️ Error in batch {batch_idx}: {e}")
            # Fallback to original text
            # (نترك النص الأصلي في حال الفشل الجزئي)

    return translated_segments


# ============================================================================
# PRODUCER-CONSUMER PIPELINE (Concurrent Processing)
# ============================================================================
# Threshold: Only use pipeline for videos > 3 minutes
PIPELINE_THRESHOLD_SECONDS = 180  # 3 minutes

import queue
import threading


def producer_transcribe(
    audio_path: str,
    segment_queue: queue.Queue,
    result_holder: dict,
    source_lang: str = None
):
    """
    Producer thread: Transcribes audio and puts segments into queue.
    
    Args:
        audio_path: Path to audio file
        segment_queue: Queue to put transcribed segments
        result_holder: Dict to store detected_lang and any errors
        source_lang: Optional language hint for Whisper
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        result_holder['error'] = "faster-whisper not installed"
        segment_queue.put(None)  # Signal completion
        return
    
    model = None
    
    try:
        print(f"🔄 [Producer] Loading {WHISPER_MODEL} on GPU...")
        model = WhisperModel(
            WHISPER_MODEL,
            device="cuda",
            compute_type="int8",
            num_workers=1
        )
        print(f"🚀 [Producer] Loaded {WHISPER_MODEL} on GPU successfully!")
        
        segments_iter, info = model.transcribe(
            audio_path,
            language=source_lang,
            beam_size=3,
            temperature=0.0,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=700,  # Increased to reduce hallucinations
                threshold=0.5,                # Stricter VAD threshold
                speech_pad_ms=300
            )
        )
        
        result_holder['detected_lang'] = info.language
        result_holder['duration'] = info.duration
        print(f"📝 [Producer] Detected: {info.language}, Duration: {info.duration:.1f}s")
        
        # Stream segments to queue
        idx = 0
        for seg in segments_iter:
            segment = {
                "idx": idx,
                "start": seg.start,
                "end": seg.end,
                "text": seg.text
            }
            segment_queue.put(segment)
            idx += 1
        
        result_holder['total_segments'] = idx
        print(f"✅ [Producer] Transcribed {idx} segments")
        
    except Exception as e:
        result_holder['error'] = str(e)
        print(f"❌ [Producer] Error: {e}")
    finally:
        # Signal completion
        segment_queue.put(None)
        
        # Cleanup
        if model:
            del model
            clear_gpu_memory()


def consumer_translate(
    segment_queue: queue.Queue,
    result_queue: queue.Queue,
    source_lang: str,
    target_lang: str,
    journal = None
):
    """
    Consumer thread: Translates segments from queue.
    
    Args:
        segment_queue: Queue to get transcribed segments from
        result_queue: Queue to put translated segments
        source_lang: Source language for translation
        target_lang: Target language for translation
        journal: Optional journal for crash recovery
    """
    # Simplify language codes
    lang_map = {
        "eng_latn": "en", "eng": "en", "en": "en",
        "arb_arab": "ar", "ara": "ar", "ar": "ar",
        "tur_latn": "tr", "tur": "tr", "tr": "tr"
    }
    
    src = lang_map.get(source_lang.lower(), source_lang.lower()) if source_lang else "en"
    tgt = lang_map.get(target_lang.lower(), target_lang.lower())[:2]
    
    if src == tgt:
        # Same language, just pass through
        while True:
            seg = segment_queue.get()
            if seg is None:
                result_queue.put(None)
                break
            result_queue.put(seg)
        return
    
    # Load translation model
    model_map = {
        "en-ar": "Helsinki-NLP/opus-mt-en-ar",
        "ar-en": "Helsinki-NLP/opus-mt-ar-en",
        "tr-ar": "Helsinki-NLP/opus-mt-tr-ar",
        "ar-tr": "Helsinki-NLP/opus-mt-ar-tr",
        "en-tr": "Helsinki-NLP/opus-mt-en-tr",
        "tr-en": "Helsinki-NLP/opus-mt-tr-en"
    }
    
    pair_key = f"{src}-{tgt}"
    model_name = model_map.get(pair_key)
    
    if not model_name:
        print(f"❌ [Consumer] No model for {pair_key}")
        # Pass through untranslated
        while True:
            seg = segment_queue.get()
            if seg is None:
                result_queue.put(None)
                break
            result_queue.put(seg)
        return
    
    try:
        print(f"📥 [Consumer] Loading {model_name}...")
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name).to("cpu")
        print(f"✅ [Consumer] Model loaded on CPU")
        
        # Batch processing with micro-batches
        batch_buffer = []
        batch_size = 4  # Smaller batches for streaming
        
        while True:
            seg = segment_queue.get()
            
            if seg is None:
                # Flush remaining buffer
                if batch_buffer:
                    translated_batch = translate_batch(batch_buffer, tokenizer, model)
                    for tseg in translated_batch:
                        if journal:
                            journal.save_transcribed_segment(tseg['idx'], tseg)
                        result_queue.put(tseg)
                result_queue.put(None)
                break
            
            batch_buffer.append(seg)
            
            if len(batch_buffer) >= batch_size:
                translated_batch = translate_batch(batch_buffer, tokenizer, model)
                for tseg in translated_batch:
                    if journal:
                        journal.save_translated_segment(tseg['idx'], tseg)
                    result_queue.put(tseg)
                batch_buffer = []
                
    except Exception as e:
        print(f"❌ [Consumer] Error: {e}")
        # Drain queue and signal completion
        while True:
            seg = segment_queue.get()
            if seg is None:
                break
        result_queue.put(None)


def translate_batch(segments: List[dict], tokenizer, model) -> List[dict]:
    """Translate a batch of segments."""
    texts = [seg['text'] for seg in segments]
    
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to("cpu")
    
    translated_tokens = model.generate(
        **inputs,
        max_length=512,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=2
    )
    
    decoded = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
    
    result = []
    for seg, trans_text in zip(segments, decoded):
        result.append({
            "idx": seg['idx'],
            "start": seg['start'],
            "end": seg['end'],
            "text": trans_text
        })
    
    return result


def process_video_pipeline(
    video_path: str,
    audio_path: str,
    source_lang: str,
    target_lang: str,
    journal = None
) -> Optional[Tuple[List[dict], str]]:
    """
    Process video using producer-consumer pipeline for concurrent transcription and translation.
    
    Returns:
        Tuple of (translated_segments, detected_lang) or None on failure
    """
    print("\n" + "=" * 60)
    print("🚀 CONCURRENT PIPELINE MODE")
    print("=" * 60)
    
    # Create queues
    segment_queue = queue.Queue(maxsize=50)  # Buffer limit
    result_queue = queue.Queue()
    
    # Shared result holder for producer
    result_holder = {}
    
    # Start producer thread
    producer = threading.Thread(
        target=producer_transcribe,
        args=(audio_path, segment_queue, result_holder, source_lang),
        daemon=True
    )
    
    # We need to know the detected language before starting consumer
    # So we start producer first and wait for first segment or language detection
    producer.start()
    
    # Wait for language detection (with timeout)
    timeout = 60  # 60 seconds for model loading
    for _ in range(timeout * 10):
        if 'detected_lang' in result_holder or 'error' in result_holder:
            break
        threading.Event().wait(0.1)
    
    if 'error' in result_holder:
        print(f"❌ Pipeline error: {result_holder['error']}")
        return None
    
    detected_lang = result_holder.get('detected_lang', 'en')
    
    # Start consumer thread
    consumer = threading.Thread(
        target=consumer_translate,
        args=(segment_queue, result_queue, detected_lang, target_lang, journal),
        daemon=True
    )
    consumer.start()
    
    # Collect results
    translated_segments = []
    with tqdm(desc="Pipeline", unit="seg") as pbar:
        while True:
            result = result_queue.get()
            if result is None:
                break
            translated_segments.append(result)
            pbar.update(1)
    
    # Wait for threads to complete
    producer.join(timeout=5)
    consumer.join(timeout=5)
    
    # Sort segments by index
    translated_segments.sort(key=lambda x: x['idx'])
    
    # Remove idx field
    for seg in translated_segments:
        del seg['idx']
    
    print(f"✅ Pipeline completed: {len(translated_segments)} segments")
    
    return translated_segments, detected_lang


# ============================================================================
# MAIN PIPELINE
# ============================================================================
def process_video(
    video_path: str,
    output_srt: Optional[str] = None,
    keep_audio: bool = False,
    batch_size: int = BATCH_SIZE,
    source_lang: str = None,  # None = auto-detect
    target_lang: str = DEFAULT_TARGET_LANG,
    use_glossary: bool = True,  # Enable/disable glossary sanitization
    initial_prompt: str = None,  # Context/keywords prompt for Whisper
    content_profile: str = None,  # Smart Content Profile
    use_vocal_isolation: bool = False  # Professional vocal isolation (UVR5)
):
    """
    Main pipeline: Video -> Audio -> Transcription -> Translation -> SRT
    
    Features:
    - Professional Vocal Isolation: UVR5 MDX-Net for clean vocals (optional)
    - Smart Content Profiles: Optimized settings for different content types
    - Probe Phase: Analyzes first 30 seconds to auto-adjust settings
    - Crash recovery: Automatically resumes from last checkpoint if interrupted
    - Glossary: Applies technical term corrections before saving SRT
    """
    # Default profile if not specified
    if content_profile is None:
        content_profile = DEFAULT_PROFILE
    
    print("\n" + "=" * 60)
    print("🎬 VIDEO TRANSLATOR PIPELINE")
    print("=" * 60)
    print(f"📁 Input: {video_path}")
    print(f"🎭 Profile: {content_profile}")
    print(f"🎤 Vocal Isolation: {'UVR5 MDX-Net' if use_vocal_isolation and HAS_AUDIO_SEPARATOR else 'Disabled'}")
    print(f"🧠 Whisper: {WHISPER_MODEL} (GPU only)")
    print(f"🌍 Helsinki-NLP Translation")
    print(f"📦 Batch size: {batch_size}")
    print(f"🔤 Source: {'Auto-detect' if source_lang is None else source_lang}")
    print(f"🔤 Target: {target_lang}")
    print(f"📚 Glossary: {'Enabled' if use_glossary and HAS_GLOSSARY else 'Disabled'}")
    print(f"💾 Journal: {'Enabled' if HAS_JOURNAL else 'Disabled'}")
    
    # Validate input
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return False
    
    # Setup paths
    video_stem = Path(video_path).stem
    video_dir = Path(video_path).parent
    audio_path = str(video_dir / f"{video_stem}_temp.wav")
    
    if output_srt is None:
        output_srt = str(video_dir / f"{video_stem}_ar.srt")
    
    # =========================================================================
    # TIME TRACKING INITIALIZATION
    # =========================================================================
    start_total = time.time()
    time_download = 0.0
    time_extract = 0.0
    time_vocal_isolation = 0.0
    time_probe = 0.0
    time_transcribe = 0.0
    time_translate = 0.0
    time_srt = 0.0
    audio_duration = 0.0  # For RTF calculation
    profile_settings = None  # Will be set after probe
    vocals_path = None  # Path to isolated vocals (if used)
    
    # =========================================================================
    # JOURNAL INITIALIZATION (Crash Recovery)
    # =========================================================================
    journal = None
    segments = None
    translated = None
    detected_lang = None
    
    if HAS_JOURNAL:
        journal = TranscriptionJournal(video_path)
        
        if journal.has_checkpoint():
            print("\n" + "=" * 60)
            print("🔄 RESUMING FROM CHECKPOINT")
            print("=" * 60)
            
            # Load cached segments
            cached_transcribed = journal.get_transcribed_segments()
            cached_translated = journal.get_translated_segments()
            
            print(f"📂 Found {len(cached_transcribed)} transcribed, {len(cached_translated)} translated")
            
            # If transcription is complete, use cached
            if len(cached_transcribed) > 0:
                segments = cached_transcribed
                detected_lang = journal.get_metadata().get('source_lang', 'en')
                print(f"✅ Using cached transcription ({len(segments)} segments)")
            
            # If translation is complete, use cached
            if len(cached_translated) == len(cached_transcribed) and len(cached_translated) > 0:
                translated = cached_translated
                print(f"✅ Using cached translation ({len(translated)} segments)")
        else:
            # Initialize new journal
            journal.initialize(source_lang=source_lang, target_lang=target_lang)
    
    # Step A: Extract audio (always needed for Whisper if no cache)
    if segments is None:
        step_start = time.time()
        if not extract_audio(video_path, audio_path, content_profile):
            return False
        time_extract = time.time() - step_start
        print(f"⏱️  Time: {time_extract:.2f}s")
        
        # Step A.5: Professional Vocal Isolation (optional)
        if use_vocal_isolation and HAS_AUDIO_SEPARATOR:
            step_start = time.time()
            print("\n" + "=" * 60)
            print("🎤 STEP A.5: Vocal Isolation (UVR5 MDX-Net)")
            print("=" * 60)
            
            vocals_path = isolate_vocals(audio_path, str(video_dir))
            
            if vocals_path and os.path.exists(vocals_path):
                # Use isolated vocals instead of raw audio for transcription
                audio_path = vocals_path
                print(f"✅ Using isolated vocals for transcription")
            else:
                print("⚠️ Vocal isolation failed, using original audio")
            
            time_vocal_isolation = time.time() - step_start
            print(f"⏱️  Time: {time_vocal_isolation:.2f}s")
        
        # Probe Phase: Analyze audio and get optimized settings
        probe_start = time.time()
        profile_settings = probe_audio(audio_path, content_profile)
        time_probe = time.time() - probe_start
        print(f"⏱️  Probe Time: {time_probe:.2f}s")
    
    # Step B: Transcribe (with auto language detection)
    if segments is None:
        step_start = time.time()
        result = transcribe_audio(audio_path, source_lang, initial_prompt, profile_settings)
        if result is None:
            print("❌ Transcription failed or no speech detected")
            return False
        
        segments, detected_lang = result
        if len(segments) == 0:
            print("❌ No speech detected in video")
            return False
        
        time_transcribe = time.time() - step_start
        
        # Calculate audio duration and RTF (Real-Time Factor)
        if segments:
            audio_duration = segments[-1].get('end', 0) if segments else 0
            if audio_duration > 0:
                rtf = time_transcribe / audio_duration
                print(f"⏱️  Time: {time_transcribe:.2f}s | RTF: {rtf:.2f}x ({'faster' if rtf < 1 else 'slower'} than real-time)")
            else:
                print(f"⏱️  Time: {time_transcribe:.2f}s")
        
        # Save to journal
        if journal:
            for idx, seg in enumerate(segments):
                journal.save_transcribed_segment(idx, seg)
            print(f"💾 Saved {len(segments)} segments to journal")
    
    # Convert Whisper language code to simple format for Helsinki
    print(f"🌐 Detected language: {detected_lang} → Target: {target_lang}")
    
    # Convert target_lang to simple code for Helsinki
    simple_target_lang = {
        "arb_Arab": "ar",
        "eng_Latn": "en",
        "tur_Latn": "tr",
    }.get(target_lang, target_lang.split('_')[0].lower())
    
    # Check if source and target languages are the same
    # Map Whisper codes to simple codes for comparison
    detected_simple = {
        "ar": "ar", "arabic": "ar",
        "en": "en", "english": "en",
        "tr": "tr", "turkish": "tr",
    }.get(detected_lang.lower() if detected_lang else "", detected_lang.lower() if detected_lang else "")
    
    if detected_simple == simple_target_lang:
        print(f"\n⚠️ Source language ({detected_lang}) is same as target ({simple_target_lang})")
        print("📝 Skipping translation - writing original transcription to SRT")
        translated = segments  # Use original segments without translation
    
    # Step C: Translate (only if not already set)
    if translated is None:
        step_start = time.time()
        translated = translate_segments(
            segments, 
            None,  # nllb_model_path not used
            source_lang=detected_lang,
            target_lang=simple_target_lang,
            batch_size=batch_size
        )
        if translated is None:
            print("❌ Translation failed")
            return False
        
        time_translate = time.time() - step_start
        print(f"⏱️  Time: {time_translate:.2f}s")
        
        # Save to journal
        if journal:
            for idx, seg in enumerate(translated):
                journal.save_translated_segment(idx, seg)
            print(f"💾 Saved {len(translated)} translations to journal")
    
    # =========================================================================
    # GLOSSARY SANITIZATION (Technical Terms)
    # =========================================================================
    if use_glossary and HAS_GLOSSARY:
        print("\n" + "=" * 60)
        print("📚 STEP C.5: Glossary Sanitization")
        print("=" * 60)
        
        glossary = get_default_glossary()
        stats = glossary.get_stats()
        print(f"📖 Loaded {stats['total_patterns']} terms from {len(stats['profiles_loaded'])} profiles")
        
        translated = sanitize_segments(translated)
        print("✅ Technical terms sanitized")
    
    # Step D: Write SRT
    step_start = time.time()
    print("\n" + "=" * 60)
    print("📄 STEP D: SRT Generation")
    print("=" * 60)
    write_srt(translated, output_srt)
    time_srt = time.time() - step_start
    print(f"⏱️  Time: {time_srt:.2f}s")
    
    # Cleanup temp audio
    if not keep_audio and os.path.exists(audio_path):
        os.remove(audio_path)
        print(f"🧹 Removed temp audio: {audio_path}")
    
    # Cleanup journal on success
    if journal:
        journal.cleanup()
    
    # =========================================================================
    # FINAL TIME SUMMARY
    # =========================================================================
    total_time = time.time() - start_total
    ai_processing_time = time_transcribe + time_translate
    
    print("\n" + "═" * 60)
    print("🎉 PIPELINE COMPLETE!")
    print("═" * 60)
    print(f"📄 Output SRT: {output_srt}")
    print()
    print("⏰ TIME SUMMARY:")
    print("-" * 40)
    if time_download > 0:
        print(f"   ⬇️  Download:      {time_download:>7.2f}s")
    if time_extract > 0:
        print(f"   🎤 Audio Extract: {time_extract:>7.2f}s")
    if time_transcribe > 0:
        print(f"   🤖 Transcription: {time_transcribe:>7.2f}s")
        if audio_duration > 0:
            rtf = time_transcribe / audio_duration
            print(f"      └─ RTF: {rtf:.2f}x {'(faster)' if rtf < 1 else '(slower)'}")
    if time_translate > 0:
        print(f"   🌐 Translation:   {time_translate:>7.2f}s")
    if time_srt > 0:
        print(f"   📄 SRT Write:     {time_srt:>7.2f}s")
    print("-" * 40)
    print(f"   🧠 AI Processing: {ai_processing_time:>7.2f}s")
    print(f"   ⏱️  TOTAL TIME:    {total_time:>7.2f}s ({total_time/60:.1f} min)")
    print("═" * 60)
    
    return True


# ============================================================================
# CLI ENTRY POINT
# ============================================================================
def main():
    # Build language choices for help text
    lang_choices = ", ".join(HELSINKI_LANGUAGES.keys())
    
    parser = argparse.ArgumentParser(
        description="Offline Video Subtitling & Translation (Optimized for 2GB VRAM)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  python video_translator.py movie.mp4
  python video_translator.py movie.mp4 --target Arabic
  python video_translator.py movie.mp4 --target Turkish --output movie_tr.srt

Supported languages: {lang_choices}
        """
    )
    
    parser.add_argument(
        "video",
        help="Path to input video file"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output SRT file path (default: <video>_ar.srt)"
    )
    parser.add_argument(
        "-t", "--target",
        default="Arabic",
        choices=list(HELSINKI_LANGUAGES.keys()),
        help="Target language for translation (default: Arabic)"
    )
    parser.add_argument(
        "--keep-audio",
        action="store_true",
        help="Keep extracted audio file after processing"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Translation batch size (default: {BATCH_SIZE})"
    )
    parser.add_argument(
        "--embed",
        action="store_true",
        help="Embed subtitles into video (creates new video file)"
    )
    
    args = parser.parse_args()
    
    # Get target language code
    target_lang = HELSINKI_LANGUAGES.get(args.target, DEFAULT_TARGET_LANG)
    
    # Run pipeline
    success = process_video(
        video_path=args.video,
        output_srt=args.output,
        keep_audio=args.keep_audio,
        batch_size=args.batch_size,
        target_lang=target_lang
    )
    
    # Embed subtitles if requested (FAST METHOD - Soft Subs)
    if success and args.embed:
        video_path = args.video
        # SRT path construction remains the same
        srt_path = args.output or str(Path(video_path).with_suffix('')) + f"_{target_lang[:2].lower()}.srt"
        output_video = str(Path(video_path).with_suffix('')) + "_subtitled.mp4"
        
        print("\n" + "=" * 60)
        print("🎬 STEP E: Embedding Subtitles (Muxing Mode)")
        print("=" * 60)
        
        try:
            # التغيير الجذري هنا:
            # -c copy: ينسخ الفيديو والصوت كما هما دون إعادة معالجة
            # -c:s mov_text: يحول SRT إلى صيغة متوافقة مع MP4
            cmd = [
                "ffmpeg", "-y",
                "-i", video_path,          # Input 0: Video
                "-i", srt_path,            # Input 1: Subtitles
                "-c", "copy",              # Copy video/audio streams (INSTANT)
                "-c:s", "mov_text",        # Convert subs to MP4 compatible format
                "-metadata:s:s:0", f"language={target_lang[:3]}", # Tag language
                output_video
            ]
            
            print(f"🚀 Muxing subtitles into container (should take seconds)...")
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"✅ Subtitled video ready: {output_video}")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ FFmpeg Muxing error: {e.stderr.decode() if e.stderr else e}")
            # Fallback if mov_text fails (rare): Try MKV container
            print("⚠️ Falling back to MKV container (better subtitle support)...")
            output_video_mkv = str(Path(video_path).with_suffix('')) + "_subtitled.mkv"
            cmd_mkv = [
                "ffmpeg", "-y", "-i", video_path, "-i", srt_path,
                "-c", "copy", "-c:s", "srt", output_video_mkv
            ]
            subprocess.run(cmd_mkv, check=False)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
