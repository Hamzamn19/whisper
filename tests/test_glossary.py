"""Tests for glossary.py — pure stdlib, no model/GPU needed."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from glossary import TechnicalGlossary, sanitize_text, sanitize_segments


def test_profiles_exist():
    g = TechnicalGlossary()
    profiles = g.get_available_profiles()
    assert "tech" in profiles
    assert "general" in profiles


def test_tech_terms_sanitized():
    g = TechnicalGlossary()
    g.load_profile("tech")
    out = g.sanitize("Install python on linux")
    assert "بايثون" in out
    assert "لينكس" in out


def test_word_boundaries_respected():
    g = TechnicalGlossary()
    g.load_profile("tech")
    # 'python' inside a longer word must not match \bpython\b
    out = g.sanitize("I like pineapples")
    assert "بايثون" not in out


def test_general_profile_additive():
    g = TechnicalGlossary()
    g.load_profile("tech")
    g.load_profile("general")
    out = g.sanitize("Check my email and python")
    assert "بريد إلكتروني" in out
    assert "بايثون" in out


def test_convenience_sanitize_text():
    out = sanitize_text("Install python on linux")
    assert "بايثون" in out


def test_sanitize_segments_preserves_timing():
    segs = [{"start": 0.0, "end": 1.5, "text": "Install python now"}]
    out = sanitize_segments(segs)
    assert len(out) == 1
    assert out[0]["start"] == 0.0
    assert out[0]["end"] == 1.5
    assert "بايثون" in out[0]["text"]
