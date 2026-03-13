"""
NVIDIA NeMo Parakeet TDT Batch Audio Transcription Tool
=====================================================
Optimized for transcribing very long audio files with high-quality subtitles.

Core Features & Strategy:
1.  Obtains both word-level and segment-level timestamps from the model simultaneously.
2.  Prioritizes using the model's native segment splits, as they are generally of the highest quality.
3.  For segments that are too long, it uses precise word-level timestamps to find natural split points.
4.  Splitting logic prioritizes natural boundaries based on punctuation (end of sentence > clause > comma).
5.  Falls back to splitting by word count only as a last resort to meet duration constraints.
6.  Features a smart pre-processing step for extremely long audio files (>60 minutes), splitting them
    at silent points before transcription to prevent memory overflow and ensure reliability.
"""

import argparse
import torch
import os
import tempfile
import subprocess
import json
import re
import math
from pathlib import Path
from nemo.collections.asr.models import ASRModel
from pydub import AudioSegment
from pydub.silence import detect_silence
import datetime
import time
import gc
import sys
import shutil
import traceback
import bisect
from typing import List, Tuple, Optional, Dict, Any

# --- Global Configuration ---

MODEL_NAME = "/kaggle/input/models/vijaykumars1228/parakeet-tdt-0-6b-v2/pytorch/pytorch/1/parakeet-tdt-0.6b-v2.nemo"
OUTPUT_DIR = "/kaggle/working/transcriptions"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Supported audio formats
AUDIO_EXTENSIONS = ['.wav', '.mp3', '.flac', '.ogg', '.m4a', '.mp4', '.opus', '.webm', '.mkv', '.avi', '.mov', '.wma', '.mka']

# Long audio processing parameters
LONG_AUDIO_THRESHOLD = 480  # 8 minutes, triggers long audio attention optimization
VERY_LONG_AUDIO_THRESHOLD = 60 * 60  # 60 minutes, triggers pre-transcription splitting
SEGMENT_TARGET_DURATION = 60 * 60  # Target duration for each split chunk (60 minutes)
SILENCE_SEARCH_WINDOW = 2 * 60  # Search window for silence around the target split time (±2 minutes)
SILENCE_THRESHOLD_DB = -25  # dBFS level to be considered silence
MIN_SILENCE_DURATION = 400  # Minimum duration of a silent pause in milliseconds

# Subtitle generation parameters (tuned for anime)
MAX_SEGMENT_DURATION = 5  # Max duration of a single subtitle line in seconds
MIN_SEGMENT_DURATION = 1  # Min duration of a single subtitle line in seconds (anime has short exclamations)
IDEAL_SEGMENT_DURATION = 3  # Ideal duration of a single subtitle line in seconds
MAX_WORDS_PER_SEGMENT = 15  # Fallback: max words if no punctuation is found for splitting

# Punctuation priority for splitting (includes both English and Chinese punctuation)
SENTENCE_ENDINGS = ('.', '!', '?', '。', '！', '？')
CLAUSE_SEPARATORS = (';', ':', '；', '：')
SOFT_SEPARATORS = (',', '，')

# --- Logging Functions ---


def print_info(message):
    """Prints an informational log message."""
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}][INFO] {message}", file=sys.stderr)


def print_error(message):
    """Prints an error log message."""
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}][ERROR] {message}", file=sys.stderr)


def print_warning(message):
    """Prints a warning log message."""
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}][WARNING] {message}", file=sys.stderr)


def print_success(message):
    """Prints a success message."""
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}][SUCCESS] {message}", file=sys.stderr)


def print_skip(message):
    """Prints a skip message."""
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}][SKIP] {message}", file=sys.stderr)


def print_debug(message):
    """Prints a debug message if the DEBUG environment variable is set."""
    if os.environ.get('DEBUG'):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}][DEBUG] {message}", file=sys.stderr)


# --- Long Audio Handling ---
class LongAudioHandler:
    """
    Handles the splitting of very long audio files and the merging of their transcription results.
    It intelligently splits at silent points to avoid cutting off words and to maintain semantic integrity.
    """

    def __init__(self, target_duration=SEGMENT_TARGET_DURATION, silence_thresh=SILENCE_THRESHOLD_DB, min_silence_len=MIN_SILENCE_DURATION, search_window=SILENCE_SEARCH_WINDOW, fast_mode=False):
        """
        Initializes the long audio handler.
        Args:
            target_duration (int): The target duration for each audio chunk in seconds.
            silence_thresh (int): The silence threshold in dBFS.
            min_silence_len (int): The minimum length of silence in milliseconds.
            search_window (int): The window in seconds to search for silence around the target split time.
            fast_mode (bool): If True, skips silence detection and splits at fixed intervals.
        """
        self.target_duration = target_duration
        self.silence_thresh = silence_thresh
        self.min_silence_len = min_silence_len
        self.search_window = search_window
        self.fast_mode = fast_mode

    def find_split_points(self, audio: AudioSegment) -> List[int]:
        """
        Finds the optimal split points in the audio, preferring silent passages.
        This optimized version only detects silence near the target split points, improving performance.
        Args:
            audio (AudioSegment): The pydub AudioSegment object.
        Returns:
            A list of split points in milliseconds.
        """
        duration_ms = len(audio)
        duration_sec = duration_ms / 1000
        # Calculate the required number of segments (using math.ceil to be safe)
        num_segments = math.ceil(duration_sec / self.target_duration)
        if num_segments <= 1:
            return []  # No splitting needed
        print_info(f"Audio is {duration_sec/60:.1f} minutes long, splitting into {num_segments} segments.")

        # Fast mode: Split at fixed intervals, skipping silence detection
        if self.fast_mode:
            print_info("Fast mode: Skipping silence detection, splitting at fixed intervals.")
            split_points = []
            for i in range(1, num_segments):
                split_point = int(i * self.target_duration * 1000)
                split_points.append(split_point)
                print_info(f"Segment {i}: Fixed split point at {split_point/1000/60:.1f} minutes.")
            return split_points

        # Standard mode: Find the best silent point near each target split time
        split_points = []
        for i in range(1, num_segments):
            target_time_ms = int(i * self.target_duration * 1000)
            # Define a search window around the target time
            window_start_ms = max(0, target_time_ms - self.search_window * 1000)
            window_end_ms = min(duration_ms, target_time_ms + self.search_window * 1000)
            print_debug(f"Detecting silence in window {window_start_ms/1000/60:.1f}-{window_end_ms/1000/60:.1f} minutes.")

            # Extracting only the window audio for silence detection greatly improves performance
            window_audio = audio[window_start_ms:window_end_ms]
            # Detect silent passages in the window
            silence_ranges = detect_silence(window_audio, min_silence_len=self.min_silence_len, silence_thresh=self.silence_thresh)

            if silence_ranges:
                # Find the longest silent passage in the window
                best_silence = max(silence_ranges, key=lambda x: x[1] - x[0])
                # Calculate the midpoint of the silence, adding the window's start offset
                silence_middle = window_start_ms + (best_silence[0] + best_silence[1]) // 2
                split_point = silence_middle
                print_info(f"Segment {i}: Found silence for splitting at {split_point/1000/60:.1f} min (target: {target_time_ms/1000/60:.1f} min).")
            else:
                # If no silence is found, fall back to the target time
                split_point = target_time_ms
                print_warning(f"Segment {i}: No silence found, forcing split at {split_point/1000/60:.1f} min.")
            split_points.append(split_point)
        return split_points

    def split_audio(self, audio_path: str, temp_dir: str, already_converted: bool = False) -> List[Dict[str, Any]]:
        """
        Splits a single audio file into multiple WAV chunks.
        Args:
            audio_path (str): Path to the source audio file.
            temp_dir (str): Path to the directory for temporary chunk files.
        Returns:
            A list of dictionaries, each containing info about a chunk.
        """
        print_info(f"Loading audio file for splitting: {os.path.basename(audio_path)}")
        # Load the audio file
        audio = AudioSegment.from_file(audio_path)
        duration_ms = len(audio)
        duration_sec = duration_ms / 1000

        # Find the split points
        split_points = self.find_split_points(audio)
        if not split_points:
            # If the audio is not long enough to be split, process it as a single segment
            print_info(f"Audio duration is {duration_sec/60:.1f} minutes, no splitting needed. Processing as a single chunk.")
            return [{'file_path': audio_path, 'start_time': 0, 'end_time': duration_sec, 'duration': duration_sec, 'segment_index': 1}]

        # Perform the split
        segments_info = []
        start_ms = 0
        for i, end_ms in enumerate(split_points + [duration_ms], 1):
            # Extract the audio chunk
            segment_audio = audio[start_ms:end_ms]
            # Save the chunk to a temporary file
            segment_filename = f"segment_{i}.wav"
            segment_path = os.path.join(temp_dir, segment_filename)
            # Export in the format required by the model: 16kHz, mono, WAV
            if not already_converted:
                segment_audio = segment_audio.set_frame_rate(16000)
                segment_audio = segment_audio.set_channels(1)
            segment_audio.export(segment_path, format="wav")
            segments_info.append({
                'file_path': segment_path,
                'start_time': start_ms / 1000,  # Convert to seconds
                'end_time': end_ms / 1000,
                'duration': (end_ms - start_ms) / 1000,
                'segment_index': i
            })
            print_success(f"Generated segment {i}: {start_ms/1000/60:.1f}-{end_ms/1000/60:.1f} minutes.")
            start_ms = end_ms
        return segments_info

    def merge_transcription_results(self, results: List[Dict[str, Any]]) -> Tuple[List[Dict], List[Dict]]:
        """
        Merges transcription results from multiple chunks, adjusting timestamps to be absolute.
        Args:
            results: A list of transcription results, one for each chunk.
        Returns:
            A tuple containing the merged list of segment timestamps and word timestamps.
        """
        merged_segments = []
        merged_words = []
        for result in results:
            offset = result['offset']
            # Adjust and merge segment timestamps
            if result.get('segments'):
                for seg in result['segments']:
                    adjusted_seg = seg.copy()
                    adjusted_seg['start'] = seg.get('start', 0) + offset
                    adjusted_seg['end'] = seg.get('end', 0) + offset
                    merged_segments.append(adjusted_seg)
            # Adjust and merge word timestamps
            if result.get('words'):
                for word in result['words']:
                    adjusted_word = word.copy()
                    adjusted_word['start'] = word.get('start', 0) + offset
                    adjusted_word['end'] = word.get('end', 0) + offset
                    merged_words.append(adjusted_word)
        return merged_segments, merged_words


# --- Subtitle Processing ---
class OptimizedSubtitleProcessor:
    """
    An optimized subtitle processor that uses NeMo's native word-level timestamps
    for precise splitting of long subtitle segments.
    """

    @staticmethod
    def process_segments(segment_timestamps: List[Dict], word_timestamps: Optional[List[Dict]] = None, max_duration: float = MAX_SEGMENT_DURATION, max_words: int = MAX_WORDS_PER_SEGMENT) -> List[Dict]:
        """
        Processes raw transcription segments to create well-formed subtitles.
        If a segment is too long, it attempts to split it using word timestamps.
        Short segments are merged to meet MIN_SEGMENT_DURATION.
        Args:
            segment_timestamps: The list of segments from NeMo.
            word_timestamps: The list of word timestamps from NeMo.
            max_duration: The maximum allowed duration for a single subtitle segment.
            max_words: The maximum words per segment (fallback splitting threshold).
        Returns:
            A list of optimized subtitle segments.
        """
        # Pre-compute word start times for efficient binary search in _split_with_words
        word_starts = [w.get('start', 0) for w in word_timestamps] if word_timestamps else []

        optimized_segments = []
        for segment in segment_timestamps:
            duration = segment.get('end', 0) - segment.get('start', 0)
            print_debug(f"Processing segment, duration {duration:.1f}s, max allowed {max_duration}s.")
            # If segment duration is acceptable, keep it as is
            if duration <= max_duration:
                optimized_segments.append(segment)
                print_debug(f"Segment duration is fine ({duration:.1f}s), keeping as is.")
            else:
                # Segment is too long and needs to be split
                print_warning(f"Found long segment ({duration:.1f}s), which needs splitting.")
                if word_timestamps:
                    # Use precise word-level timestamps for splitting
                    split_segments = OptimizedSubtitleProcessor._split_with_words(segment, word_timestamps, max_duration, max_words, word_starts)
                    print_success("Splitting completed using word-level timestamps.")
                else:
                    # Fallback to text-based splitting if no word timestamps are available
                    print_warning("No word-level timestamps available, falling back to text-based splitting.")
                    split_segments = OptimizedSubtitleProcessor._split_by_text(segment, max_duration)
                optimized_segments.extend(split_segments)
                print_debug(f"Split into {len(split_segments)} segments.")

        # Merge segments that are too short to be readable
        optimized_segments = OptimizedSubtitleProcessor._merge_short_segments(optimized_segments, max_duration)
        return optimized_segments

    @staticmethod
    def _merge_short_segments(segments: List[Dict], max_duration: float) -> List[Dict]:
        """
        Merges subtitle segments shorter than MIN_SEGMENT_DURATION with adjacent segments,
        while respecting MAX_SEGMENT_DURATION. Targets IDEAL_SEGMENT_DURATION when possible.
        This prevents subtitle flicker from very short lines (common in anime).
        Args:
            segments: The list of subtitle segments.
            max_duration: The maximum allowed duration for a merged segment.
        Returns:
            A list of segments with short ones merged into neighbors.
        """
        if not segments or len(segments) < 2:
            return segments

        merged = []
        i = 0
        merge_count = 0
        while i < len(segments):
            current = segments[i].copy()
            current_duration = current.get('end', 0) - current.get('start', 0)

            # If the current segment is too short, try to merge with the next segment
            if current_duration < MIN_SEGMENT_DURATION and i + 1 < len(segments):
                next_seg = segments[i + 1]
                combined_duration = next_seg.get('end', 0) - current.get('start', 0)

                # Only merge if the combined duration stays within limits
                if combined_duration <= max_duration:
                    current_text = current.get('segment', '').strip()
                    next_text = next_seg.get('segment', '').strip()
                    merged_text = f"{current_text} {next_text}".strip()
                    current = {
                        'start': current.get('start', 0),
                        'end': next_seg.get('end', 0),
                        'segment': merged_text
                    }
                    merge_count += 1
                    i += 2  # Skip the next segment since it's been merged
                    merged.append(current)
                    continue

            merged.append(current)
            i += 1

        if merge_count > 0:
            print_info(f"Merged {merge_count} short segments (< {MIN_SEGMENT_DURATION}s) with adjacent segments.")
        return merged

    @staticmethod
    def _split_with_words(segment: Dict, word_timestamps: List[Dict], max_duration: float, max_words: int = MAX_WORDS_PER_SEGMENT, word_starts: Optional[List[float]] = None) -> List[Dict]:
        """
        Splits a single long segment with precision using word-level timestamps.
        Args:
            segment: The long segment to split.
            word_timestamps: The complete list of word timestamps for the audio.
            max_duration: The maximum allowed duration for a segment.
            max_words: The maximum words per segment (fallback threshold).
            word_starts: Pre-computed list of word start times for binary search.
        Returns:
            A list of smaller, split segments.
        """
        # Find all words that belong to this specific segment using binary search
        segment_start = segment.get('start', 0)
        segment_end = segment.get('end', float('inf'))

        # Use binary search for efficient word lookup (O(log N) instead of O(N))
        if word_starts is None:
            word_starts = [w.get('start', 0) for w in word_timestamps]
        lo = bisect.bisect_left(word_starts, segment_start - 0.1)

        segment_words = []
        for word_info in word_timestamps[lo:]:
            word_start = word_info.get('start', 0)
            if word_start > segment_end + 0.1:
                break  # Past the segment boundary, stop searching
            word_end = word_info.get('end', word_start)
            if word_end <= segment_end + 0.1:
                segment_words.append(word_info)
        if not segment_words:
            print_warning("Could not find corresponding word timestamps for the segment.")
            return OptimizedSubtitleProcessor._split_by_text(segment, max_duration)

        # Intelligently split using word timestamps
        split_segments = []
        current_segment = {'words': [], 'start': segment_words[0].get('start', 0)}
        for i, word_info in enumerate(segment_words):
            word_text = word_info.get('word', word_info.get('char', ''))
            word_end = word_info.get('end', word_info.get('start', 0))

            # Add the word to the current potential segment
            current_segment['words'].append(word_info)
            current_duration = word_end - current_segment['start']

            # Determine if we should split after this word
            should_split = False
            split_reason = ""
            # Condition 1: Exceeded max duration, must split
            if current_duration > max_duration:
                should_split = True
                split_reason = f"Exceeded max duration {current_duration:.1f}s"
            # Condition 2: Approaching max duration and a natural sentence break occurs
            elif current_duration > max_duration * 0.7:  # 70% is a reasonable threshold
                if any(word_text.rstrip().endswith(punct) for punct in SENTENCE_ENDINGS):
                    should_split = True
                    split_reason = f"Sentence ending at {current_duration:.1f}s"
                elif current_duration > max_duration * 0.8 and any(word_text.rstrip().endswith(punct) for punct in CLAUSE_SEPARATORS):
                    should_split = True
                    split_reason = f"Clause separator at {current_duration:.1f}s"
                elif current_duration > max_duration * 0.9 and any(word_text.rstrip().endswith(punct) for punct in SOFT_SEPARATORS):
                    should_split = True
                    split_reason = f"Soft separator at {current_duration:.1f}s"
            # Condition 3: Exceeded max word count (last resort)
            elif len(current_segment['words']) >= max_words:
                should_split = True
                split_reason = f"Reached max words ({max_words})"

            # Perform the split
            if should_split and len(current_segment['words']) > 0:
                print_debug(f"Split reason: {split_reason}")
                segment_text = ' '.join([w.get('word', w.get('char', '')) for w in current_segment['words']])
                split_segments.append({'start': current_segment['start'], 'end': current_segment['words'][-1].get('end', current_segment['start']), 'segment': segment_text.strip()})

                # Prepare the next segment if there are more words
                if i < len(segment_words) - 1:
                    next_word = segment_words[i + 1]
                    current_segment = {'words': [], 'start': next_word.get('start', 0)}

        # Add the final remaining segment
        if current_segment['words']:
            segment_text = ' '.join([w.get('word', w.get('char', '')) for w in current_segment['words']])
            split_segments.append({'start': current_segment['start'], 'end': current_segment['words'][-1].get('end', current_segment['start']), 'segment': segment_text.strip()})
        return split_segments

    @staticmethod
    def _split_by_text(segment: Dict, max_duration: float) -> List[Dict]:
        """
        Splits a segment based on its text content (fallback method).
        Args:
            segment: The long segment to split.
            max_duration: The maximum allowed duration.
        Returns:
            A list of smaller, split segments.
        """
        text = segment.get('segment', '')
        start_time = segment.get('start', 0)
        end_time = segment.get('end', start_time)
        duration = end_time - start_time
        if not text:
            return [segment]

        # Calculate how many pieces we need to split it into
        num_segments = max(1, int(duration / max_duration) + 1)
        # First, try to split by punctuation
        sentences = OptimizedSubtitleProcessor._split_by_punctuation(text)
        if len(sentences) >= num_segments:
            # We have enough sentences to group them
            sentences_per_group = max(1, len(sentences) // num_segments)
            split_texts = []
            for i in range(0, len(sentences), sentences_per_group):
                group_sentences = sentences[i:i + sentences_per_group]
                if group_sentences:
                    split_texts.append(' '.join(group_sentences))
        else:
            # Not enough sentences, fall back to splitting by word count
            words = text.split()
            words_per_segment = max(1, len(words) // num_segments)
            split_texts = []
            for i in range(0, len(words), words_per_segment):
                segment_words = words[i:i + words_per_segment]
                if segment_words:
                    split_texts.append(' '.join(segment_words))
        result = []
        total_length = sum(len(s) for s in split_texts)
        if total_length == 0:
            return [segment]
        current_time = start_time
        for split_text in split_texts:
            text_ratio = len(split_text) / total_length
            segment_duration = duration * text_ratio
            result.append({'start': current_time, 'end': min(current_time + segment_duration, end_time), 'segment': split_text.strip()})
            current_time += segment_duration
        return result

    @staticmethod
    def _split_by_punctuation(text: str) -> List[str]:
        """
        Splits text into sentences based on punctuation.
        Args:
            text: The text to split.
        Returns:
            A list of sentences.
        """
        # Regex to split by punctuation while keeping the punctuation
        pattern = r'([.!?;:。！？；：])\s*'
        parts = re.split(pattern, text)
        sentences = []
        current = ""
        # Recombine text parts with their trailing punctuation
        for i, part in enumerate(parts):
            if i % 2 == 0:
                current = part
            else:
                if current:
                    sentences.append(current + part)
                current = ""
        if current:
            sentences.append(current)  # Add the last part if it has no trailing punctuation

        # Filter out empty sentences and merge very short ones
        filtered_sentences = []
        min_sentence_length = 10
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if filtered_sentences and len(sentence) < min_sentence_length:
                filtered_sentences[-1] += ' ' + sentence
            else:
                filtered_sentences.append(sentence)
        return filtered_sentences if filtered_sentences else [text]


# --- Audio & System Utilities ---
def get_audio_info(audio_path: str) -> Optional[float]:
    """Gets audio duration using ffprobe."""
    try:
        cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams', audio_path]
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore', check=True, timeout=120)
        info = json.loads(result.stdout)
        duration = float(info['format'].get('duration', 0))
        return duration
    except subprocess.TimeoutExpired:
        print_error(f"ffprobe timed out for {audio_path}")
        return None
    except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError) as e:
        print_error(f"Failed to get audio info for {audio_path}: {str(e)}")
        return None


def process_audio_with_ffmpeg(audio_path: str, temp_dir: str) -> Tuple[Optional[str], Optional[float]]:
    """
    Converts audio to a standardized WAV format (16kHz, mono) required by the model.
    Args:
        audio_path: Path to the source audio file.
        temp_dir: Path to the temporary directory for output files.
    """
    try:
        print_info(f"Processing audio file: {os.path.basename(audio_path)}")
        duration_sec = get_audio_info(audio_path)
        if duration_sec is None:
            # Lightweight fallback: use ffprobe with simpler output to get duration
            try:
                cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                       '-of', 'default=noprint_wrappers=1:nokey=1', audio_path]
                result = subprocess.run(cmd, capture_output=True, text=True,
                                        encoding='utf-8', errors='ignore', timeout=120)
                if result.returncode == 0 and result.stdout.strip():
                    duration_sec = float(result.stdout.strip())
                else:
                    print_error(f"Could not determine audio duration for {audio_path}")
                    return None, None
            except (subprocess.TimeoutExpired, ValueError) as e:
                print_error(f"Failed to probe audio duration: {str(e)}")
                return None, None
        print_info(f"Audio duration: {format_time(duration_sec)}")

        os.makedirs(temp_dir, exist_ok=True)

        # Create a unique path for the processed WAV file
        temp_filename = f"processed_{os.getpid()}_{time.time()}.wav"
        processed_path = os.path.join(temp_dir, temp_filename)

        # ffmpeg command to convert audio (with timeout to prevent hangs on corrupted files)
        cmd = ['ffmpeg', '-i', audio_path, '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le', '-y', '-loglevel', 'error', processed_path]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore', timeout=600)
        except subprocess.TimeoutExpired:
            print_error(f"ffmpeg timed out while processing {os.path.basename(audio_path)}")
            return None, None
        if result.returncode != 0:
            print_error(f"ffmpeg processing failed: {result.stderr}")
            return None, None

        if not os.path.exists(processed_path):
            print_error("ffmpeg failed to generate output file.")
            return None, None

        print_info("Audio processing successful.")
        return processed_path, duration_sec
    except Exception as e:
        print_error(f"An error occurred during audio processing: {str(e)}")
        return None, None


def cleanup_temp_directory(temp_dir: str):
    """Removes the specified temporary directory and its contents."""
    if temp_dir and os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
            print_debug(f"Deleted temporary directory: {temp_dir}")
        except Exception as e:
            print_warning(f"Could not delete temporary directory {temp_dir}: {str(e)}")


def format_time(seconds):
    """Formats seconds into HH:MM:SS."""
    return str(datetime.timedelta(seconds=seconds)).split('.')[0]


def format_srt_time(seconds):
    """Formats seconds into the SRT timestamp format (HH:MM:SS,ms)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def create_srt(segments):
    """Generates SRT content from a list of segments."""
    srt_lines = []
    subtitle_index = 1
    for segment in segments:
        text = segment.get('segment', '').strip()
        if not text:
            continue
        start_time = format_srt_time(segment.get('start', 0))
        end_time = format_srt_time(segment.get('end', 0))
        srt_lines.append(f"{subtitle_index}")
        srt_lines.append(f"{start_time} --> {end_time}")
        srt_lines.append(text)
        srt_lines.append("")
        subtitle_index += 1
    return "\n".join(srt_lines)


def merge_duplicate_segments(segments: List[Dict]) -> List[Dict]:
    """Removes identical subtitle segments (same text and timestamps)."""
    if not segments:
        return []
    merged = []
    seen = set()
    for segment in segments:
        text = segment.get('segment', '').strip()
        start = segment.get('start', 0)
        end = segment.get('end', 0)
        if not text:
            continue
        # Create a unique ID for the segment, rounding timestamps to avoid float precision issues
        unique_id = (text, round(start, 3), round(end, 3))
        if unique_id not in seen:
            seen.add(unique_id)
            merged.append(segment)
        else:
            print_debug(f"Removing duplicate subtitle: {text[:50]}... [{start:.3f}-{end:.3f}]")
    removed_count = len(segments) - len(merged)
    if removed_count > 0:
        print_info(f"Removed {removed_count} identical subtitles.")
    return merged


def validate_subtitle_timestamps(segments: List[Dict], min_gap: float = 0.05) -> List[Dict]:
    """
    Validates and fixes subtitle timestamps to ensure monotonic ordering
    with no overlaps and a minimum gap between consecutive segments.
    This prevents subtitle flashing/flicker in players.
    Args:
        segments: List of subtitle segments.
        min_gap: Minimum gap in seconds between consecutive segments (default: 50ms).
    Returns:
        The validated and fixed list of segments.
    """
    if not segments or len(segments) < 2:
        return segments

    fixed = [segments[0].copy()]
    fixes_count = 0

    for i in range(1, len(segments)):
        current = segments[i].copy()
        prev = fixed[-1]
        prev_end = prev.get('end', 0)
        curr_start = current.get('start', 0)

        # Fix overlap: current starts before previous ends
        if curr_start < prev_end:
            fixed[-1]['end'] = curr_start - min_gap
            fixes_count += 1
            print_debug(f"Fixed overlap at {curr_start:.3f}s: adjusted previous end to {fixed[-1]['end']:.3f}s")
        # Fix insufficient gap between segments
        elif curr_start - prev_end < min_gap:
            fixed[-1]['end'] = curr_start - min_gap
            fixes_count += 1
            print_debug(f"Added gap at {curr_start:.3f}s: adjusted previous end to {fixed[-1]['end']:.3f}s")

        fixed.append(current)

    if fixes_count > 0:
        print_info(f"Fixed {fixes_count} timestamp overlaps/gaps in subtitles.")
    return fixed


# --- Main Transcriber Class ---
class BatchTranscriber:
    """
    Orchestrates the entire batch transcription process, from loading the model
    to processing files and generating subtitles.
    """

    def __init__(self, model_name: str = MODEL_NAME, device: str = DEVICE):
        """Initializes the BatchTranscriber."""
        self.model_name = model_name
        self.device = device
        self.model = None
        self.subtitle_processor = OptimizedSubtitleProcessor()
        self.long_audio_handler = LongAudioHandler()
        self.temp_dir = tempfile.mkdtemp(prefix="nemo_transcribe_")
        print_debug(f"Created temporary directory: {self.temp_dir}")
        self.stats = {
            'processed': 0,
            'skipped': 0,
            'failed': 0,
            'total_duration': 0,
            'total_time': 0,
            'total_segments': 0,
            'split_segments': 0,
            'word_level_splits': 0,
            'long_audio_splits': 0
        }

    def load_model(self):
        """Loads the ASR model into memory."""
        if self.model is not None:
            return True
        try:
            print_info(f"Loading Parakeet TDT model on {self.device.upper()}...")
            if self.model_name.endswith('.nemo'):
                # Load from a local .nemo file (e.g. Kaggle models)
                print_info(f"Loading model from local .nemo file: {self.model_name}")
                self.model = ASRModel.restore_from(restore_path=self.model_name)
            else:
                # Download from NVIDIA NGC / HuggingFace
                self.model = ASRModel.from_pretrained(model_name=self.model_name)
            self.model.eval()
            self.model = self.model.to(self.device)
            print_success(f"Model successfully loaded to {self.device.upper()}!")
            print_info("Verifying model's timestamp support...")
            return True
        except Exception as e:
            print_error(f"Failed to load model: {str(e)}")
            return False

    @staticmethod
    def _extract_hypothesis_data(hypothesis, fallback_duration: float = 0) -> Tuple[str, List[Dict], List[Dict]]:
        """
        Extracts text, segment timestamps, and word timestamps from a NeMo hypothesis object.
        Consolidates the duplicated extraction logic into a single method.
        Args:
            hypothesis: The NeMo hypothesis object (or a plain string).
            fallback_duration: Duration to use for the fallback segment if no timestamps are found.
        Returns:
            A tuple of (full_text, segment_timestamps, word_timestamps).
        """
        full_text = ""
        segment_timestamps = []
        word_timestamps = []

        if hasattr(hypothesis, 'text'):
            full_text = hypothesis.text
        elif isinstance(hypothesis, str):
            full_text = hypothesis

        if hasattr(hypothesis, 'timestamp'):
            timestamp_dict = hypothesis.timestamp
            if isinstance(timestamp_dict, dict) and 'segment' in timestamp_dict:
                segment_timestamps = timestamp_dict['segment']
                print_info(f"Obtained {len(segment_timestamps)} segment timestamps.")
            if isinstance(timestamp_dict, dict) and 'word' in timestamp_dict:
                word_timestamps = timestamp_dict['word']
                print_info(f"Obtained {len(word_timestamps)} word-level timestamps.")

        # If no text but segments exist, reconstruct text from segments
        if not full_text and segment_timestamps:
            full_text = ' '.join([seg.get('segment', '') for seg in segment_timestamps])

        # If no segments but text exists, create a fallback segment
        if not segment_timestamps and full_text:
            print_warning("No segment timestamps found, creating a default segment.")
            segment_timestamps = [{'start': 0, 'end': fallback_duration, 'segment': full_text}]

        return full_text, segment_timestamps, word_timestamps

    def transcribe_segments_separately(self, segments_info: List[Dict], apply_long_audio_settings: bool = True) -> List[Dict]:
        """
        Transcribes a list of audio file segments one by one.
        Args:
            segments_info: A list of dictionaries with info about each audio chunk.
            apply_long_audio_settings: Whether to apply the long audio attention model.
        Returns:
            A list of transcription results for each chunk.
        """
        results = []
        total_segments = len(segments_info)
        for seg_info in segments_info:
            segment_index = seg_info['segment_index']
            segment_path = seg_info['file_path']
            segment_start = seg_info['start_time']
            print_info(f"[{segment_index}/{total_segments}] Transcribing segment {segment_index}...")
            try:
                if apply_long_audio_settings:
                    try:
                        if hasattr(self.model, 'change_attention_model'):
                            self.model.change_attention_model("rel_pos_local_attn", [256, 256])
                            print_debug("Applied long audio attention optimization.")
                    except Exception as e:
                        print_warning(f"Error applying long audio settings: {str(e)}")

                with torch.no_grad():
                    hypotheses = self.model.transcribe([segment_path], timestamps=True)

                if not hypotheses or len(hypotheses) == 0:
                    print_error(f"Transcription failed for segment {segment_index}.")
                    continue

                hypothesis = hypotheses[0]
                full_text, segment_timestamps, word_timestamps = BatchTranscriber._extract_hypothesis_data(
                    hypothesis, fallback_duration=seg_info['duration'])

                results.append({'segments': segment_timestamps, 'words': word_timestamps, 'text': full_text, 'offset': segment_start})
                print_success(f"Finished transcribing segment {segment_index}.")
                del hypotheses
                del hypothesis
                if torch.cuda.is_available(): 
                    torch.cuda.empty_cache()

            except Exception as e:
                print_error(f"Error transcribing segment {segment_index}: {str(e)}")
                continue
            finally:
                if apply_long_audio_settings:
                    try:
                        if hasattr(self.model, 'change_attention_model'):
                            self.model.change_attention_model("rel_pos")
                    except Exception:
                        pass

                try:
                    # Only delete files that are in the temp directory (not original source files)
                    if os.path.exists(segment_path) and os.path.abspath(segment_path).startswith(os.path.abspath(self.temp_dir)):
                        os.remove(segment_path)
                        print_debug(f"Deleted segment file: {segment_path}")
                except Exception:
                    pass
        return results

    def transcribe_single(self, audio_path: str, output_path: str) -> bool:
        """
        Processes a single audio file from start to finish.
        Args:
            audio_path: Path to the input audio file.
            output_path: Path for the output SRT file.
        Returns:
            True if successful, False otherwise.
        """
        start_time = time.time()
        temp_audio_path = None
        try:
            temp_audio_path, duration_sec = process_audio_with_ffmpeg(audio_path, self.temp_dir)
            if temp_audio_path is None:
                return False

            # Decide processing strategy based on audio duration
            if duration_sec > VERY_LONG_AUDIO_THRESHOLD:
                # --- Strategy for VERY LONG audio: Split before transcribing ---
                print_warning(f"Audio is longer than {VERY_LONG_AUDIO_THRESHOLD/60:.0f} minutes, enabling splitting process...")
                self.stats['long_audio_splits'] += 1

                segments_info = self.long_audio_handler.split_audio(temp_audio_path, self.temp_dir, already_converted=True)
                transcription_results = self.transcribe_segments_separately(segments_info, apply_long_audio_settings=True)
                if not transcription_results:
                    print_error("All segment transcriptions failed.")
                    return False

                merged_segments, merged_words = self.long_audio_handler.merge_transcription_results(transcription_results)
                print_info(f"Merging complete: {len(merged_segments)} segments, {len(merged_words)} words.")
                optimized_segments = self.subtitle_processor.process_segments(merged_segments, merged_words if merged_words else None, MAX_SEGMENT_DURATION)
            else:
                # --- Strategy for REGULAR and LONG audio: Transcribe in one go ---
                long_audio_settings_applied = False
                if duration_sec > LONG_AUDIO_THRESHOLD:
                    try:
                        print_info(f"Audio is longer than {LONG_AUDIO_THRESHOLD/60:.1f} minutes, applying optimized settings...")
                        if hasattr(self.model, 'change_attention_model'):
                            self.model.change_attention_model("rel_pos_local_attn", [256, 256])
                            long_audio_settings_applied = True
                            print_info("Applied long audio attention optimization.")
                    except Exception as e:
                        print_warning(f"Error applying long audio settings: {str(e)}")

                try:
                    print_info("Starting transcription (with word and segment-level timestamps)...")
                    with torch.no_grad():
                        hypotheses = self.model.transcribe([temp_audio_path], timestamps=True)

                    if not hypotheses or len(hypotheses) == 0:
                        print_error("Transcription failed or produced no output.")
                        return False

                    hypothesis = hypotheses[0]
                    full_text, segment_timestamps, word_timestamps = BatchTranscriber._extract_hypothesis_data(
                        hypothesis, fallback_duration=duration_sec)

                    del hypotheses
                    del hypothesis
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    if not full_text:
                        print_error("Could not get transcribed text.")
                        return False

                    optimized_segments = self.subtitle_processor.process_segments(segment_timestamps, word_timestamps, MAX_SEGMENT_DURATION)
                    del segment_timestamps
                    del word_timestamps
                    del full_text

                finally:
                    if long_audio_settings_applied:
                        try:
                            if hasattr(self.model, 'change_attention_model'):
                                self.model.change_attention_model("rel_pos")
                                print_debug("Restored default attention model settings.")
                        except Exception:
                            pass

            # --- Finalize and save subtitle file ---
            deduplicated_segments = merge_duplicate_segments(optimized_segments)
            validated_segments = validate_subtitle_timestamps(deduplicated_segments)
            srt_content = create_srt(validated_segments)
            if not srt_content.strip():
                print_warning("Generated subtitle content is empty.")
                return False

            # Atomic write: write to temp file first, then rename to prevent partial files
            temp_output_path = output_path + '.tmp'
            with open(temp_output_path, 'w', encoding='utf-8') as f:
                f.write(srt_content)
            os.replace(temp_output_path, output_path)

            # Update and print stats for this file
            processing_time = time.time() - start_time
            self.stats['processed'] += 1
            self.stats['total_duration'] += duration_sec
            self.stats['total_time'] += processing_time
            self.stats['total_segments'] += len(optimized_segments)
            max_seg_dur = max((s.get('end', 0) - s.get('start', 0) for s in optimized_segments), default=0)

            print_success(f"Transcription complete: {os.path.basename(audio_path)} ({len(optimized_segments)} segments, longest: {max_seg_dur:.1f}s)")
            print_success(f"Media Duration: {format_time(duration_sec)}\n"
                          f"Transcription Time: {processing_time:.1f} seconds\n"
                          f"Processing Speed: {duration_sec/processing_time if processing_time > 0 else 0:.2f}x real-time\n")

            del optimized_segments
            del srt_content
            return True
        except Exception as e:
            print_error(f"An error occurred while transcribing {os.path.basename(audio_path)}: {str(e)}")
            print_debug(traceback.format_exc())
            self.stats['failed'] += 1
            return False
        finally:
            # Clean up temporary files and memory
            if temp_audio_path and os.path.exists(temp_audio_path):
                try:
                    os.remove(temp_audio_path)
                    print_debug(f"Deleted temporary file: {temp_audio_path}")
                except Exception:
                    pass

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if os.environ.get('DEBUG'):
                    allocated = torch.cuda.memory_allocated() / 1024**2
                    reserved = torch.cuda.memory_reserved() / 1024**2
                    print_debug(f"VRAM Usage: {allocated:.1f}MB allocated, {reserved:.1f}MB reserved.")

    def process_directory(self, directory: str, output_dir: str = OUTPUT_DIR, extensions: List[str] = None, skip_existing: bool = True) -> None:
        """Finds and processes all audio files in a given directory."""
        if extensions is None: 
            extensions = AUDIO_EXTENSIONS

        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        print_info(f"Output directory: {os.path.abspath(output_dir)}")

        audio_files = []
        for ext in extensions:
            audio_files.extend(Path(directory).glob(f'*{ext}'))
            audio_files.extend(Path(directory).glob(f'*{ext.upper()}'))

        audio_files = sorted(set(audio_files))
        if not audio_files:
            print_warning(f"No audio files found in {directory}")
            return

        total_files = len(audio_files)
        print_info(f"Found {total_files} audio files to process.")
        print("=" * 60)

        for idx, audio_file in enumerate(audio_files, 1):
            # Save SRT files to the output directory instead of next to the input
            output_file = Path(output_dir) / audio_file.with_suffix('.srt').name
            print(f"\n[{idx}/{total_files}] Processing: {audio_file.name}")

            if skip_existing and output_file.exists():
                file_size = output_file.stat().st_size
                if file_size > 0:
                    print_skip(f"Subtitle file already exists and is not empty: {output_file.name}")
                    self.stats['skipped'] += 1
                    continue
                else:
                    print_warning(f"Subtitle file exists but is empty, will regenerate: {output_file.name}")

            success = self.transcribe_single(str(audio_file), str(output_file))
            if not success:
                print_error(f"Failed to transcribe: {audio_file.name}")

        self.print_statistics()

    def print_statistics(self):
        """Prints a summary of the batch processing results."""
        print("\n" + "=" * 60)
        print("Batch Transcription Summary")
        print("=" * 60)
        print(f"Successfully processed: {self.stats['processed']} files")
        print(f"Skipped:              {self.stats['skipped']} files")
        print(f"Failed:               {self.stats['failed']} files")
        if self.stats['processed'] > 0:
            print(f"Total audio duration:   {format_time(self.stats['total_duration'])}")
            print(f"Total processing time:  {format_time(self.stats['total_time'])}")
            avg_speed = self.stats['total_duration'] / self.stats['total_time'] if self.stats['total_time'] > 0 else 0
            print(f"Average speed:          {avg_speed:.2f}x real-time")
            avg_time = self.stats['total_time'] / self.stats['processed']
            print(f"Average time per file:  {avg_time:.1f} seconds")
            print(f"Total segments created: {self.stats['total_segments']}")
            if self.stats['split_segments'] > 0:
                print(f"Segments from splits:   {self.stats['split_segments']}")
            if self.stats['word_level_splits'] > 0:
                print(f"Word-level splits used: {self.stats['word_level_splits']}")
            if self.stats['long_audio_splits'] > 0:
                print(f"Very long audios split: {self.stats['long_audio_splits']} files")
            avg_segments = self.stats['total_segments'] / self.stats['processed']
            print(f"Average segments per file: {avg_segments:.1f}")
        print("=" * 60)

    def cleanup(self):
        """Releases resources held by the transcriber."""
        if self.model is not None:
            del self.model
            self.model = None
        cleanup_temp_directory(self.temp_dir)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


# --- Main Execution ---
def check_dependencies():
    """Checks for necessary external dependencies."""
    dependencies_ok = True
    try:
        import pydub
        print_debug("pydub is installed.")
    except ImportError:
        print_error("pydub not found. Please install it: pip install pydub")
        dependencies_ok = False

    try:
        # Check if ffmpeg/libav is available for pydub to use
        test_audio = AudioSegment.silent(duration=100)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            test_audio.export(tmp.name, format="wav")
        print_debug("Audio processing backend seems to be working.")
    except Exception as e:
        print_warning(f"Audio processing backend might have issues: {str(e)}")
        print_info("Please ensure that FFmpeg is installed and accessible in your system's PATH.")
    return dependencies_ok


def main():
    """Main entry point of the script."""
    global MAX_SEGMENT_DURATION, MAX_WORDS_PER_SEGMENT, MIN_SEGMENT_DURATION

    parser = argparse.ArgumentParser(description="Batch transcribe audio files to SRT subtitles (with smart splitting for long audio).", formatter_class=argparse.RawDescriptionHelpFormatter, epilog="""
Optimization Strategy:
  1. Prefers the model's native 'segment' splits, which are usually high quality.
  2. For segments that exceed the max duration, it uses precise 'word' timestamps to split them.
  3. Splitting Priority: End-of-sentence > Clause separator > Soft separator (comma) > Word count.
  4. For very long audio (>60 min), it pre-splits the file at silent points to prevent memory issues.

Default Parameters (Tuned for Anime):
  - Max Segment Duration: 5 seconds
  - Min Segment Duration: 1 second
  - Max Words Limit: 15 words (fallback threshold)
  - Long Audio Threshold: 8 minutes (applies model optimization)
  - Very Long Audio Threshold: 60 minutes (triggers pre-splitting)

Usage Examples:
  # Transcribe all audio in a directory
  %(prog)s /path/to/audio/directory

  # Use CUDA and process only .mp4 and .mkv files
  %(prog)s ~/videos --device cuda --extensions .mp4 .mkv

  # Force re-transcription of all files, overwriting existing .srt files
  %(prog)s . --no-skip

  # Adjust segmenting rules for a podcast
  %(prog)s ./podcasts --max-segment 10 --max-words 40
        """)

    parser.add_argument("directory", help="Path to the directory containing audio files.")
    parser.add_argument("-d", "--device", choices=["cuda", "cpu", "auto"], default="auto", help="Inference device (default: auto).")
    parser.add_argument("-e", "--extensions", nargs="+", help=f"File extensions to process (default: all supported formats).")
    parser.add_argument("--no-skip", action="store_true", help="Overwrite and re-transcribe if subtitle files already exist.")
    parser.add_argument("-o", "--output", default=OUTPUT_DIR, help=f"Output directory for SRT files (default: {OUTPUT_DIR}).")
    parser.add_argument("-m", "--model", default=MODEL_NAME, help=f"Name or path of the model to use (default: {MODEL_NAME}).")
    parser.add_argument("--max-segment", type=int, default=MAX_SEGMENT_DURATION, help=f"Maximum subtitle segment duration in seconds (default: {MAX_SEGMENT_DURATION}).")
    parser.add_argument("--min-segment", type=float, default=MIN_SEGMENT_DURATION, help=f"Minimum subtitle segment duration in seconds; shorter segments are merged (default: {MIN_SEGMENT_DURATION}).")
    parser.add_argument("--max-words", type=int, default=MAX_WORDS_PER_SEGMENT, help=f"Maximum words per segment, used as a fallback (default: {MAX_WORDS_PER_SEGMENT}).")
    parser.add_argument("--debug", action="store_true", help="Enable detailed debug output.")
    args = parser.parse_args()

    if args.debug:
        os.environ['DEBUG'] = '1'

    MAX_SEGMENT_DURATION = args.max_segment
    MIN_SEGMENT_DURATION = args.min_segment
    MAX_WORDS_PER_SEGMENT = args.max_words

    if not check_dependencies():
        sys.exit(1)

    if not os.path.isdir(args.directory):
        print_error(f"Directory not found: {args.directory}")
        sys.exit(1)

    global DEVICE
    if args.device == "auto":
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        DEVICE = args.device

    extensions = args.extensions if args.extensions else AUDIO_EXTENSIONS
    extensions = [ext if ext.startswith('.') else f'.{ext}' for ext in extensions]

    print("=" * 60)
    print("Batch Audio Transcription Tool (NeMo with Word-level Timestamps)")
    print("=" * 60)
    print(f"Input Directory:       {os.path.abspath(args.directory)}")
    print(f"Output Directory:      {os.path.abspath(args.output)}")
    print(f"Device:                {DEVICE.upper()}")
    print(f"Model:                 {args.model}")
    print(f"File Types:            {', '.join(extensions)}")
    print(f"Skip Existing:         {'No' if args.no_skip else 'Yes'}")
    print(f"Max Segment Duration:  {MAX_SEGMENT_DURATION} seconds")
    print(f"Min Segment Duration:  {MIN_SEGMENT_DURATION} seconds")
    print(f"Max Words/Segment:     {MAX_WORDS_PER_SEGMENT} words")
    print(f"Very Long Audio Split: >{VERY_LONG_AUDIO_THRESHOLD/60:.0f} minutes")
    print(f"Debug Mode:            {'On' if args.debug else 'Off'}")
    print("=" * 60)

    transcriber = BatchTranscriber(model_name=args.model, device=DEVICE)
    try:
        if not transcriber.load_model():
            print_error("Could not load the transcription model. Exiting.")
            sys.exit(1)
        transcriber.process_directory(directory=args.directory, output_dir=args.output, extensions=extensions, skip_existing=not args.no_skip)
    except KeyboardInterrupt:
        print("\n\nUser interrupted the process. Printing final statistics.")
        transcriber.print_statistics()
    except Exception as e:
        print_error(f"A critical error occurred during processing: {str(e)}")
        traceback.print_exc()
    finally:
        print_info("Cleaning up resources...")
        transcriber.cleanup()
        print_info("Done!")


if __name__ == "__main__":
    main()
