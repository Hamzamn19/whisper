"""Config consistency tests — parse sources as text (no GPU/model imports)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _read(name):
    return (ROOT / name).read_text(encoding="utf-8")


def test_dictation_uses_large_v3_float16():
    src = _read("smart_whisper_google_optimized.py")
    assert 'MODEL_SIZE = "large-v3"' in src
    assert 'COMPUTE_TYPE = "float16"' in src


def test_translator_matches_dictation_settings():
    src = _read("video_translator.py")
    assert 'WHISPER_MODEL = "large-v3"' in src
    assert 'compute_type="int8"' not in src


def test_cuda_lib_path_covers_venv():
    src = _read("smart_whisper_google_optimized.py")
    assert ".venv" in src and "nvidia" in src


def test_all_scripts_compile():
    import py_compile
    for name in ["smart_whisper_google_optimized.py", "video_translator.py",
                 "video_translator_gui.py", "glossary.py", "journal.py"]:
        py_compile.compile(str(ROOT / name), doraise=True)
