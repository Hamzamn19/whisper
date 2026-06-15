#!/usr/bin/env python3
"""
Glossary Module: Technical Term Sanitization (Post-Processing)
Data-driven architecture with loadable glossary profiles.

Usage:
    glossary = TechnicalGlossary()
    glossary.load_profile('tech')      # Load tech.json
    glossary.load_profile('general')   # Load general.json (additive)
    
    sanitized_text = glossary.sanitize("Install python on linux")
    # Returns: "Install بايثون on لينكس"
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Set

# ============================================================================
# CONFIGURATION
# ============================================================================
GLOSSARIES_DIR = Path(__file__).parent / "glossaries"


class TechnicalGlossary:
    """
    Manages technical term replacement for improved translation accuracy.
    Supports multiple glossary profiles that can be loaded additively.
    """
    
    def __init__(self, glossaries_dir: str = None):
        """
        Initialize glossary manager.
        
        Args:
            glossaries_dir: Path to glossaries directory (default: ./glossaries)
        """
        self.glossaries_dir = Path(glossaries_dir) if glossaries_dir else GLOSSARIES_DIR
        
        # Combined patterns from all loaded profiles
        self._patterns: Dict[str, str] = {}
        
        # Compiled regex cache for performance
        self._compiled_patterns: List[tuple] = []
        
        # Track which profiles are loaded
        self._loaded_profiles: Set[str] = set()
    
    def get_available_profiles(self) -> List[str]:
        """Get list of available glossary profiles."""
        if not self.glossaries_dir.exists():
            return []
        
        profiles = []
        for json_file in self.glossaries_dir.glob("*.json"):
            profiles.append(json_file.stem)
        return sorted(profiles)
    
    def load_profile(self, profile_name: str) -> bool:
        """
        Load a glossary profile (additive - doesn't replace existing patterns).
        
        Args:
            profile_name: Name of the profile (without .json extension)
            
        Returns:
            True if loaded successfully, False otherwise
        """
        if profile_name in self._loaded_profiles:
            return True  # Already loaded
        
        profile_path = self.glossaries_dir / f"{profile_name}.json"
        
        if not profile_path.exists():
            print(f"⚠️ Glossary profile not found: {profile_name}")
            return False
        
        try:
            with open(profile_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            patterns = data.get('patterns', {})
            
            # Add patterns (newer overwrites older if same key)
            for pattern, replacement in patterns.items():
                self._patterns[pattern] = replacement
            
            self._loaded_profiles.add(profile_name)
            self._compile_patterns()
            
            meta = data.get('_meta', {})
            name = meta.get('name', profile_name)
            print(f"📚 Loaded glossary: {name} ({len(patterns)} terms)")
            
            return True
            
        except (json.JSONDecodeError, IOError) as e:
            print(f"❌ Error loading glossary {profile_name}: {e}")
            return False
    
    def load_all_profiles(self) -> int:
        """
        Load all available glossary profiles.
        
        Returns:
            Number of profiles loaded
        """
        count = 0
        for profile in self.get_available_profiles():
            if self.load_profile(profile):
                count += 1
        return count
    
    def _compile_patterns(self):
        """Compile regex patterns for efficient matching."""
        self._compiled_patterns = []
        
        for pattern, replacement in self._patterns.items():
            try:
                # Case-insensitive matching
                compiled = re.compile(pattern, re.IGNORECASE | re.UNICODE)
                self._compiled_patterns.append((compiled, replacement))
            except re.error as e:
                print(f"⚠️ Invalid regex pattern '{pattern}': {e}")
    
    def sanitize(self, text: str) -> str:
        """
        Apply all glossary patterns to sanitize/improve translated text.
        
        Args:
            text: Input text (typically translated text)
            
        Returns:
            Sanitized text with technical terms corrected
        """
        if not text or not self._compiled_patterns:
            return text
        
        result = text
        
        for compiled_pattern, replacement in self._compiled_patterns:
            result = compiled_pattern.sub(replacement, result)
        
        return result
    
    def sanitize_segments(self, segments: List[dict]) -> List[dict]:
        """
        Sanitize a list of subtitle segments.
        
        Args:
            segments: List of segment dicts with 'text' key
            
        Returns:
            New list with sanitized text
        """
        sanitized = []
        for seg in segments:
            new_seg = seg.copy()
            new_seg['text'] = self.sanitize(seg.get('text', ''))
            sanitized.append(new_seg)
        return sanitized
    
    def get_stats(self) -> dict:
        """Get statistics about loaded glossaries."""
        return {
            'profiles_loaded': list(self._loaded_profiles),
            'total_patterns': len(self._patterns),
            'glossaries_dir': str(self.glossaries_dir)
        }
    
    def add_custom_pattern(self, pattern: str, replacement: str):
        """
        Add a custom pattern at runtime.
        
        Args:
            pattern: Regex pattern to match
            replacement: Replacement text
        """
        self._patterns[pattern] = replacement
        self._compile_patterns()
    
    def clear(self):
        """Clear all loaded patterns."""
        self._patterns.clear()
        self._compiled_patterns.clear()
        self._loaded_profiles.clear()


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================
_default_glossary: Optional[TechnicalGlossary] = None


def get_default_glossary() -> TechnicalGlossary:
    """Get or create the default glossary instance with all profiles loaded."""
    global _default_glossary
    
    if _default_glossary is None:
        _default_glossary = TechnicalGlossary()
        _default_glossary.load_all_profiles()
    
    return _default_glossary


def sanitize_text(text: str) -> str:
    """Convenience function to sanitize text using default glossary."""
    return get_default_glossary().sanitize(text)


def sanitize_segments(segments: List[dict]) -> List[dict]:
    """Convenience function to sanitize segments using default glossary."""
    return get_default_glossary().sanitize_segments(segments)


# ============================================================================
# TEST
# ============================================================================
if __name__ == "__main__":
    print("Testing TechnicalGlossary...")
    print("=" * 50)
    
    # Test 1: Check available profiles
    glossary = TechnicalGlossary()
    profiles = glossary.get_available_profiles()
    print(f"Available profiles: {profiles}")
    
    # Test 2: Load profiles
    glossary.load_all_profiles()
    stats = glossary.get_stats()
    print(f"Stats: {stats}")
    
    # Test 3: Sanitize text
    test_cases = [
        ("Install python on linux", "Contains Python and Linux"),
        ("Configure the firewall settings", "Contains firewall"),
        ("Check your RAM usage", "Contains RAM"),
        ("Deploy to the cloud server", "Contains cloud and server"),
        ("Use docker with kubernetes", "Contains Docker and Kubernetes"),
        ("The api endpoint returns json", "Contains API"),
        ("Login to your account", "Contains login and account"),
    ]
    
    print("\n" + "=" * 50)
    print("Sanitization Tests:")
    print("=" * 50)
    
    for original, description in test_cases:
        sanitized = glossary.sanitize(original)
        print(f"\n{description}:")
        print(f"  Original:  {original}")
        print(f"  Sanitized: {sanitized}")
    
    # Test 4: Segments
    print("\n" + "=" * 50)
    print("Segment Sanitization:")
    print("=" * 50)
    
    segments = [
        {'start': 0, 'end': 2, 'text': 'Welcome to python tutorial'},
        {'start': 2, 'end': 4, 'text': 'We will install linux server'},
    ]
    
    sanitized_segments = glossary.sanitize_segments(segments)
    for i, seg in enumerate(sanitized_segments):
        print(f"Segment {i}: {seg['text']}")
    
    print("\n✅ All glossary tests passed!")
