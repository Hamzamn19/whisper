#!/usr/bin/env python3
"""
Journal Module: Crash-Resilient State Management
Implements a journaling system similar to ext4 for crash recovery.

Usage:
    journal = TranscriptionJournal(video_path)
    if journal.has_checkpoint():
        segments, translated = journal.load_checkpoint()
        # Resume from checkpoint
    else:
        # Start fresh
    
    # During processing:
    journal.save_transcribed_segment(idx, segment)
    journal.save_translated_segment(idx, segment)
    
    # On success:
    journal.cleanup()
"""

import hashlib
import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# ============================================================================
# JOURNAL CONFIGURATION
# ============================================================================
JOURNAL_DIR = Path.home() / ".cache" / "video_translator" / "journals"
HASH_CHUNK_SIZE = 65536  # 64KB chunks for hashing large files


class TranscriptionJournal:
    """
    Manages persistent state for crash recovery.
    Uses JSONL format for append-only writes (atomic operations).
    """
    
    def __init__(self, video_path: str):
        """
        Initialize journal for a video file.
        
        Args:
            video_path: Path to the video file being processed
        """
        self.video_path = Path(video_path).resolve()
        self.video_hash = self._compute_hash()
        
        # Create journal directory
        JOURNAL_DIR.mkdir(parents=True, exist_ok=True)
        
        # Journal file path: journals/<video_hash>.journal.jsonl
        self.journal_path = JOURNAL_DIR / f"{self.video_hash[:16]}.journal.jsonl"
        
        # In-memory cache
        self._transcribed_segments: Dict[int, dict] = {}
        self._translated_segments: Dict[int, dict] = {}
        self._metadata: Optional[dict] = None
        
        # Load existing journal if present
        if self.journal_path.exists():
            self._load_journal()
    
    def _compute_hash(self) -> str:
        """
        Compute MD5 hash of the video file for unique identification.
        Uses chunked reading for memory efficiency with large files.
        
        Returns:
            Hexadecimal hash string
        """
        hasher = hashlib.md5()
        
        try:
            with open(self.video_path, 'rb') as f:
                # Also include file size and modification time for faster "identity check"
                stat = os.fstat(f.fileno())
                hasher.update(f"{stat.st_size}:{stat.st_mtime}".encode())
                
                # Read first and last chunks for speed (good enough for identity)
                # Full file hashing would be more accurate but too slow for large videos
                first_chunk = f.read(HASH_CHUNK_SIZE)
                hasher.update(first_chunk)
                
                # Seek to end and read last chunk
                if stat.st_size > HASH_CHUNK_SIZE * 2:
                    f.seek(-HASH_CHUNK_SIZE, 2)
                    last_chunk = f.read(HASH_CHUNK_SIZE)
                    hasher.update(last_chunk)
                    
        except IOError as e:
            # Fallback to just filename + size
            hasher.update(str(self.video_path).encode())
            
        return hasher.hexdigest()
    
    def _load_journal(self):
        """Load existing journal file into memory."""
        try:
            with open(self.journal_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        entry = json.loads(line)
                        entry_type = entry.get('type')
                        
                        if entry_type == 'meta':
                            # Verify hash matches
                            if entry.get('hash') != self.video_hash:
                                print(f"⚠️ Journal hash mismatch, starting fresh")
                                self._clear_cache()
                                return
                            self._metadata = entry
                            
                        elif entry_type == 'transcribed':
                            idx = entry.get('idx')
                            self._transcribed_segments[idx] = {
                                'start': entry['start'],
                                'end': entry['end'],
                                'text': entry['text']
                            }
                            
                        elif entry_type == 'translated':
                            idx = entry.get('idx')
                            self._translated_segments[idx] = {
                                'start': entry['start'],
                                'end': entry['end'],
                                'text': entry['text']
                            }
                            
                    except json.JSONDecodeError:
                        continue  # Skip malformed lines
                        
            print(f"📂 Loaded journal: {len(self._transcribed_segments)} transcribed, "
                  f"{len(self._translated_segments)} translated")
                  
        except IOError:
            pass  # No journal exists yet
    
    def _clear_cache(self):
        """Clear in-memory cache."""
        self._transcribed_segments.clear()
        self._translated_segments.clear()
        self._metadata = None
    
    def _append_entry(self, entry: dict):
        """Append a single entry to the journal file (atomic write)."""
        with open(self.journal_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            f.flush()
            os.fsync(f.fileno())  # Force write to disk
    
    def has_checkpoint(self) -> bool:
        """Check if a valid checkpoint exists for this video."""
        return self._metadata is not None and len(self._transcribed_segments) > 0
    
    def initialize(self, source_lang: str = None, target_lang: str = None):
        """
        Initialize a new journal session.
        Call this at the start of processing.
        """
        if self._metadata is None:
            self._metadata = {
                'type': 'meta',
                'hash': self.video_hash,
                'video': str(self.video_path),
                'source_lang': source_lang,
                'target_lang': target_lang
            }
            self._append_entry(self._metadata)
            print(f"📝 Journal initialized: {self.journal_path.name}")
    
    def save_transcribed_segment(self, idx: int, segment: dict):
        """
        Save a transcribed segment to journal.
        
        Args:
            idx: Segment index
            segment: Dict with 'start', 'end', 'text'
        """
        if idx not in self._transcribed_segments:
            entry = {
                'type': 'transcribed',
                'idx': idx,
                'start': segment['start'],
                'end': segment['end'],
                'text': segment['text']
            }
            self._append_entry(entry)
            self._transcribed_segments[idx] = segment
    
    def save_translated_segment(self, idx: int, segment: dict):
        """
        Save a translated segment to journal.
        
        Args:
            idx: Segment index
            segment: Dict with 'start', 'end', 'text' (translated)
        """
        if idx not in self._translated_segments:
            entry = {
                'type': 'translated',
                'idx': idx,
                'start': segment['start'],
                'end': segment['end'],
                'text': segment['text']
            }
            self._append_entry(entry)
            self._translated_segments[idx] = segment
    
    def get_transcribed_segments(self) -> List[dict]:
        """Get all transcribed segments in order."""
        return [
            self._transcribed_segments[i] 
            for i in sorted(self._transcribed_segments.keys())
        ]
    
    def get_translated_segments(self) -> List[dict]:
        """Get all translated segments in order."""
        return [
            self._translated_segments[i] 
            for i in sorted(self._translated_segments.keys())
        ]
    
    def get_transcribed_count(self) -> int:
        """Get number of transcribed segments."""
        return len(self._transcribed_segments)
    
    def get_translated_count(self) -> int:
        """Get number of translated segments."""
        return len(self._translated_segments)
    
    def get_resume_point(self) -> Tuple[int, int]:
        """
        Get resume points for transcription and translation.
        
        Returns:
            Tuple of (transcription_start_idx, translation_start_idx)
        """
        trans_idx = max(self._transcribed_segments.keys()) + 1 if self._transcribed_segments else 0
        transl_idx = max(self._translated_segments.keys()) + 1 if self._translated_segments else 0
        return trans_idx, transl_idx
    
    def cleanup(self):
        """Delete journal file after successful completion."""
        try:
            if self.journal_path.exists():
                self.journal_path.unlink()
                print(f"🧹 Journal cleaned up: {self.journal_path.name}")
            self._clear_cache()
        except IOError as e:
            print(f"⚠️ Could not delete journal: {e}")
    
    def get_metadata(self) -> Optional[dict]:
        """Get journal metadata (source/target language, etc.)."""
        return self._metadata


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def list_pending_journals() -> List[Path]:
    """List all pending (incomplete) journal files."""
    if not JOURNAL_DIR.exists():
        return []
    return list(JOURNAL_DIR.glob("*.journal.jsonl"))


def cleanup_old_journals(max_age_days: int = 7):
    """Remove journals older than specified days."""
    import time
    
    if not JOURNAL_DIR.exists():
        return
    
    cutoff = time.time() - (max_age_days * 86400)
    
    for journal_file in JOURNAL_DIR.glob("*.journal.jsonl"):
        try:
            if journal_file.stat().st_mtime < cutoff:
                journal_file.unlink()
                print(f"🧹 Removed old journal: {journal_file.name}")
        except IOError:
            pass


# ============================================================================
# TEST
# ============================================================================
if __name__ == "__main__":
    # Simple test
    import tempfile
    
    # Create a dummy video file
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(b"dummy video content for testing")
        test_video = f.name
    
    try:
        print("Testing TranscriptionJournal...")
        
        # Test 1: Create journal
        journal = TranscriptionJournal(test_video)
        journal.initialize(source_lang="en", target_lang="ar")
        
        # Test 2: Save segments
        journal.save_transcribed_segment(0, {'start': 0.0, 'end': 2.5, 'text': 'Hello world'})
        journal.save_transcribed_segment(1, {'start': 2.5, 'end': 5.0, 'text': 'How are you'})
        
        assert journal.get_transcribed_count() == 2, "Should have 2 transcribed segments"
        
        # Test 3: Save translations
        journal.save_translated_segment(0, {'start': 0.0, 'end': 2.5, 'text': 'مرحبا بالعالم'})
        
        assert journal.get_translated_count() == 1, "Should have 1 translated segment"
        
        # Test 4: Reload journal (simulate restart)
        journal2 = TranscriptionJournal(test_video)
        assert journal2.has_checkpoint(), "Should have checkpoint"
        assert journal2.get_transcribed_count() == 2, "Should reload 2 transcribed"
        assert journal2.get_translated_count() == 1, "Should reload 1 translated"
        
        resume_trans, resume_transl = journal2.get_resume_point()
        assert resume_trans == 2, f"Should resume transcription at idx 2, got {resume_trans}"
        assert resume_transl == 1, f"Should resume translation at idx 1, got {resume_transl}"
        
        # Test 5: Cleanup
        journal2.cleanup()
        assert not journal.journal_path.exists(), "Journal should be deleted"
        
        print("✅ All journal tests passed!")
        
    finally:
        os.unlink(test_video)
