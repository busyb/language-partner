from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict, AsyncGenerator, Optional
import re

import torchaudio
import whisper
from piper import PiperVoice
import io
import wave
from fastapi import UploadFile
import base64
import tempfile
import tempfile
import subprocess
import mlx_whisper
import numpy as np
import torch
from llama_cpp import Llama
from faster_whisper import WhisperModel  # Best for M1

from audio.audio_decoder import AudioDecoder

SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2
PCM_FRAME_MS = 50

PCM_CHUNK_BYTES = int(SAMPLE_RATE * BYTES_PER_SAMPLE * (PCM_FRAME_MS / 1000))

class TTSManager:
    def __init__(self):
        # TTSManager is in /project/utility/
        # Audio is in /project/audio/ (go UP one dir)
        self.whisper_model = WhisperModel(
                        "base",        # model name: tiny, base, small, medium, large
                        device="cpu",  # M1 CPU; can also try "mps" for Apple GPU acceleration
                        compute_type="int8"  # low-memory, fast
                    )

        # self.whisper_model = self.whisper_model = whisper.load_model("turbo")
        BASE_DIR = Path(__file__).parent.parent  # utility/ → project/
        self.voice_paths = {
            "es": BASE_DIR / "audio" / "piper_models" / "es_ES-sharvard-medium.onnx",
            "fr": BASE_DIR / "audio" / "piper_models" / "fr_FR-siwis-medium.onnx",
            "zh": BASE_DIR / "audio" / "piper_models" / "zh_CN-huayan-medium.onnx",
            "en": BASE_DIR / "audio" / "piper_models" / "en_US-amy-medium.onnx"
        }
        self.start_decoder()
        self.whisper_buffer = np.array([], dtype=np.float32)

        self.voices = {}
        self._preload_all()

    def _preload_all(self):
        for lang, path in self.voice_paths.items():
            print(f"Loading {path}")  # Debug
            self.voices[lang] = PiperVoice.load(str(path))
        print("✅ ALL voices preloaded!")

    async def warmup(self):
        """🚀 Warmup for FastAPI lifespan - test all models"""
        print("🔥 Warming up Whisper...")

        # Test Whisper with dummy audio (fast)
        dummy_audio = io.BytesIO()
        with wave.open(dummy_audio, 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(b'\x00\x00' * 160)  # 10ms silence

        dummy_audio.seek(0)
        with tempfile.NamedTemporaryFile(suffix=".wav") as temp_wav:
            temp_wav.write(dummy_audio.read())
            temp_wav.flush()
            self.whisper_model.transcribe(temp_wav.name, language=None)

        # Test Piper voices (synthesize short phrase)
        print("🔥 Warming up Piper voices...")
        for lang in self.voices:
            voice = self.voices[lang]
            dummy_wav = io.BytesIO()
            with wave.open(dummy_wav, 'wb') as wav:
                voice.synthesize_wav("test", wav)  # 1-word warmup
            print(f"✅ {lang} voice warmed up")

        print("✅ TTSManager warmup complete!")

    # def feed_pcm_to_whisper(self, pcm_bytes: bytes = b"") -> str:
    #     # Add new PCM bytes
    #     if pcm_bytes:
    #         if not hasattr(self, "_pcm_buffer"):
    #             self._pcm_buffer = bytearray()
    #         self._pcm_buffer.extend(pcm_bytes)
    #
    #     if not self._pcm_buffer or len(self._pcm_buffer) == 0:
    #         return ""
    #
    #     # Convert accumulated PCM buffer → float32 numpy array
    #     audio_float = np.frombuffer(self._pcm_buffer, dtype=np.int16).astype(np.float32) / 32768.0
    #
    #     # Transcribe
    #     segments, _ = self.whisper_model.transcribe(audio_float, language=None, beam_size=5, vad_filter=True)
    #
    #     # Flush buffer
    #     self._pcm_buffer = bytearray()
    #
    #     return "".join([seg.text for seg in segments]).strip()

    LANG_NAME = {
        "en": "English",
        "es": "Spanish",
        "fr": "French",
        "zh": "Chinese",
    }
    @staticmethod
    def get_initial_prompt(target_lang: str) -> str:
        """
        Generate initial prompt for Whisper transcription.
        English is considered native, target_lang is spoken.
        """
        target_name = TTSManager.LANG_NAME.get(target_lang, "Unknown")
        return f"{target_name} is spoken but English might be mixed in"


    def feed_pcm_to_whisper(self, pcm_bytes: Optional[bytes] = None, language: Optional[str] = None) -> str:
        """
        Incremental feed to Whisper.
        - pcm_bytes: bytes of PCM16 audio chunk
            - None → finalize stream
            - b"" → ignore
        - Returns: partial transcript if available, otherwise empty string
        """
        if not hasattr(self, "_pcm_buffer"):
            self._pcm_buffer = bytearray()

        # End-of-stream
        if pcm_bytes is None:
            if not self._pcm_buffer:
                return ""
            audio_float = np.frombuffer(self._pcm_buffer, dtype=np.int16).astype(np.float32) / 32768.0
            segments, _ = self.whisper_model.transcribe(audio_float, language=language, beam_size=5, vad_filter=True, initial_prompt = None)
            self._pcm_buffer = bytearray()
            return "".join([seg.text for seg in segments]).strip()

        # Ignore empty bytes
        if len(pcm_bytes) == 0:
            return ""

        # Append PCM chunk
        self._pcm_buffer.extend(pcm_bytes)

        # Optional: do VAD-based partial transcription every N frames
        # or every chunk if small
        # For simplicity, return empty string here; only finalize on None
        return ""

    def text_to_speech_base64(self, text: str, main_lang: str) -> str:
        voice = self.voices[main_lang]

        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            voice.synthesize_wav(text, wav_file)
        wav_buffer.seek(0)
        return base64.b64encode(wav_buffer.read()).decode()

    import io
    import torchaudio
    import numpy as np

    def transcribe_audio(self, audio: UploadFile, lang: str):
        """
        Transcribe browser-uploaded audio (UploadFile) using Faster-Whisper.
        Safe for webm/mp4/wav/mp3.
        Returns segments compatible with PronunciationGrader.
        """

        # Preserve original extension when possible
        suffix = Path(audio.filename).suffix or ".webm"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            temp_path = f.name
            data = audio.file.read()

            if len(data) < 1000:
                raise ValueError("Audio file too small or empty")
            f.write(data)  # UploadFile.file is sync-safe here

        try:
            segments, info = self.whisper_model.transcribe(
                temp_path,
                word_timestamps=True,
                language=None if lang == "zh" else lang,
                task="transcribe",
                beam_size=5,
                vad_filter=True
            )

            formatted_segments = []
            full_text = []

            for segment in segments:
                seg = {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                    "words": []
                }

                if segment.words:
                    for w in segment.words:
                        clean_word = w.word.strip()
                        seg["words"].append({
                            "word": clean_word,
                            "start": w.start,
                            "end": w.end,
                            "probability": w.probability
                        })
                        full_text.append(clean_word)

                formatted_segments.append(seg)

            return {
                "segments": formatted_segments,
                "text": " ".join(full_text)
            }

        finally:
            os.unlink(temp_path)

    def generate_audio(self, text: str, lang: str) -> bytes:
        """
        Generate WAV bytes from text using Piper voice.
        Works with FastAPI FileResponse.
        """
        # Pick the voice for the requested language
        voice = self.voices.get(lang)
        if voice is None:
            raise ValueError(f"No voice configured for language: {lang}")

        # Create temp WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            temp_path = temp_wav.name

        try:
            # Synthesize audio directly to file path
            voice.synthesize_wav(text, temp_path)

            # Read back as bytes
            with open(temp_path, "rb") as f:
                wav_bytes = f.read()

        finally:
            # Ensure cleanup
            if os.path.exists(temp_path):
                os.unlink(temp_path)

        return wav_bytes

    def text_to_pcm_chunks(self, text: str, main_lang: str, chunk_ms=20):
        """Convert text to PCM16 bytes in small chunks"""
        voice = self.voices[main_lang]

        # Synthesize full WAV in memory
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            voice.synthesize_wav(text, wav_file)
        wav_buffer.seek(0)

        # Read PCM frames
        with wave.open(wav_buffer, "rb") as wav_file:
            n_channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            framerate = wav_file.getframerate()
            n_frames_per_chunk = int(framerate * (chunk_ms / 1000))

            while True:
                frames = wav_file.readframes(n_frames_per_chunk)
                if not frames:
                    break
                # Send as base64 for WebSocket
                yield base64.b64encode(frames).decode()

    def text_to_pcm_chunks_streaming(
            self, text: str, main_lang: str, chunk_ms: int = 20, window_ms: int = 300
    ):
        """
        Stream Piper TTS audio in phrase-level chunks with small PCM windows to reduce choppiness.

        - main_lang: language key for self.voices
        - chunk_ms: size of PCM sub-chunks read from WAV
        - window_ms: duration (ms) of PCM window to accumulate before sending
        Yields base64-encoded PCM16 chunks suitable for WebSocket streaming.
        """
        voice = self.voices[main_lang]

        phrase_buffer = ""
        word_count = 0
        MAX_WORDS = 8  # flush after this many words
        pcm_window = b""
        frames_in_window = 0
        framerate = 24000  # adjust to your TTS sample rate
        frames_per_window = int(framerate * (window_ms / 1000))

        for token in self.tokenize_for_tts(text):
            phrase_buffer += token
            word_count += 1

            # Flush phrase on punctuation or max words
            if token in ".!?," or word_count >= MAX_WORDS:
                # Synthesize phrase to in-memory WAV
                wav_buffer = io.BytesIO()
                with wave.open(wav_buffer, "wb") as wav_file:
                    voice.synthesize_wav(phrase_buffer, wav_file)
                wav_buffer.seek(0)

                # Read PCM frames in small chunks
                with wave.open(wav_buffer, "rb") as wav_file:
                    framerate = wav_file.getframerate()
                    n_frames_per_chunk = int(framerate * (chunk_ms / 1000))

                    while True:
                        frames = wav_file.readframes(n_frames_per_chunk)
                        if not frames:
                            break
                        pcm_window += frames
                        frames_in_window += len(frames) // 2  # PCM16 → 2 bytes per frame

                        # Send PCM window if full
                        if frames_in_window >= frames_per_window:
                            yield base64.b64encode(pcm_window).decode()
                            pcm_window = b""
                            frames_in_window = 0

                phrase_buffer = ""
                word_count = 0

        # Flush remaining phrase
        if phrase_buffer.strip():
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, "wb") as wav_file:
                voice.synthesize_wav(phrase_buffer, wav_file)
            wav_buffer.seek(0)

            with wave.open(wav_buffer, "rb") as wav_file:
                n_frames_per_chunk = int(framerate * (chunk_ms / 1000))
                while True:
                    frames = wav_file.readframes(n_frames_per_chunk)
                    if not frames:
                        break
                    pcm_window += frames
                    frames_in_window += len(frames) // 2
                    if frames_in_window >= frames_per_window:
                        yield base64.b64encode(pcm_window).decode()
                        pcm_window = b""
                        frames_in_window = 0

        # Send any remaining PCM
        if pcm_window:
            yield base64.b64encode(pcm_window).decode()


    # def text_to_pcm_chunks_streaming(self, text: str, main_lang: str, chunk_ms: int = 20):
    #     """
    #     Stream TTS audio for the full text in small PCM chunks (base64-encoded).
    #     Returns a generator of base64 PCM16 audio.
    #
    #     This no longer flushes per token, only full text phrases.
    #     """
    #     voice = self.voices[main_lang]
    #
    #     # Generate WAV in memory
    #     wav_buffer = io.BytesIO()
    #     with wave.open(wav_buffer, "wb") as wav_file:
    #         voice.synthesize_wav(text, wav_file)  # your existing method
    #     wav_buffer.seek(0)
    #
    #     # Yield PCM16 chunks
    #     with wave.open(wav_buffer, "rb") as wav_file:
    #         framerate = wav_file.getframerate()
    #         n_frames_per_chunk = int(framerate * (chunk_ms / 1000))
    #         while True:
    #             frames = wav_file.readframes(n_frames_per_chunk)
    #             if not frames:
    #                 break
    #             yield base64.b64encode(frames).decode()

    def _pcm_chunks_from_wav_bytes(self, wav_bytes: bytes, chunk_ms: int):
        """
        Helper to convert in-memory WAV bytes to PCM16 base64 chunks.
        """
        wav_buffer = io.BytesIO(wav_bytes)
        with wave.open(wav_buffer, "rb") as wav_file:
            framerate = wav_file.getframerate()
            n_frames_per_chunk = int(framerate * (chunk_ms / 1000))

            while True:
                frames = wav_file.readframes(n_frames_per_chunk)
                if not frames:
                    break
                yield base64.b64encode(frames).decode()

    async def webm_to_wav_and_transcribe(
            self,
            file: UploadFile,
            practicing_lang: str,
            native_lang: str
    ) -> str:
        """
        Convert WebM audio to WAV and transcribe using mlx_whisper.
        Optimized for Apple Silicon with mixed-language speech.
        """

        with tempfile.NamedTemporaryFile(suffix=".webm") as webm, \
                tempfile.NamedTemporaryFile(suffix=".wav") as wav:
            # 1️⃣ Save uploaded WebM
            webm.write(await file.read())
            webm.flush()

            # 2️⃣ Convert to mono 16kHz WAV (Whisper requirement)
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i", webm.name,
                    "-ar", "16000",
                    "-ac", "1",
                    wav.name,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True
            )

            # 3️⃣ Bilingual transcription prompt
            # (Used as a soft bias, NOT enforced)
            initial_prompt = (
                f"You are transcribing spoken audio. "
                f"The speaker may switch between {practicing_lang} and {native_lang}. "
                f"No other languages are expected. "
                f"Transcribe exactly what is spoken, preserving each language."
            )

            # 4️⃣ Transcribe with mlx_whisper
            result = mlx_whisper.transcribe(
                wav.name,
                # language=None,  # IMPORTANT: allow mixed language
                # prompt=initial_prompt,  # mlx_whisper uses `prompt`, not `initial_prompt`
                temperature=0.0  # more deterministic, fewer hallucinations
            )

            return result["text"].strip()

    # async def webm_to_wav_and_transcribe(self, file: UploadFile, practicing_lang: str, native_lang: str) -> str:
    #     """Convert WebM to WAV and transcribe with dynamic bilingual prompt."""
    #     with tempfile.NamedTemporaryFile(suffix=".webm") as webm, \
    #             tempfile.NamedTemporaryFile(suffix=".wav") as wav:
    #         # Save WebM
    #         webm.write(await file.read())
    #         webm.flush()
    #
    #         # Convert to WAV
    #         subprocess.run(
    #             ["ffmpeg", "-y", "-i", webm.name, "-ar", "16000", "-ac", "1", wav.name],
    #             stdout=subprocess.DEVNULL,
    #             stderr=subprocess.DEVNULL,
    #             check=True
    #         )
    #
    #         # Construct prompt
    #         transcription_prompt = (
    #             f"You are a professional transcriber, fluent in {practicing_lang} and {native_lang}.\n"
    #             "You are listening to a recording in which a person is potentially speaking both "
    #             f"{practicing_lang} and {native_lang}, and no other languages.\n"
    #             "They may be speaking only one of these languages. They may have a strong accent.\n"
    #             "You are to transcribe utterances of each language accordingly."
    #         )
    #
    #         # Transcribe with Whisper
    #         result = self.whisper_model.transcribe(
    #             wav.name,
    #             language=None,  # auto-detect
    #             initial_prompt=transcription_prompt
    #         )
    #
    #         return result["text"]

    # stream methods

    async def transcribe_audio_chunk(self, webm_bytes: bytes, lang: str = "es") -> str:
        """
        Convert WebM chunk → WAV → Whisper tiny transcription (300ms)
        For 100ms audio chunks from MediaRecorder
        """
        try:
            # STEP 1: WebM → PCM WAV in memory (50ms)
            wav_buffer = io.BytesIO()

            # Use pydub (pip install pydub) - handles WebM → WAV perfectly
            from pydub import AudioSegment

            # Load WebM bytes
            audio_segment = AudioSegment.from_file(io.BytesIO(webm_bytes), format="webm")

            # Export as 16kHz mono WAV (Whisper optimal)
            audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)
            audio_segment.export(wav_buffer, format="wav")
            wav_buffer.seek(0)

            # STEP 2: Whisper tiny transcription (250ms)
            result = self.whisper_model.transcribe(
                wav_buffer,
                language=lang,  # "es", "en", "zh"
                # fp16=False,  # CPU friendly
                temperature=0.0  # Deterministic
            )

            transcript = result["text"].strip()

            # Skip silence/noise (<2 words)
            if len(transcript.split()) < 2:
                return ""

            return transcript

        except Exception as e:
            print(f"Transcription error: {e}")
            return ""

    def start_decoder(self):
        """Start FFmpeg decoder for streaming WebM → PCM"""
        self.decoder = AudioDecoder()
        self.decoder.start()

    def stop_decoder(self):
        """Stop FFmpeg decoder"""
        if self.decoder:
            self.decoder.stop()
            self.decoder = None

    def feed_webm_chunk(self, webm_bytes: bytes) -> bytes:
        """
        Convert a full WebM bytes buffer → PCM16 bytes in memory for Whisper.
        Avoids using FFmpeg subprocess in streaming mode.
        """
        from pydub import AudioSegment
        import io

        audio = AudioSegment.from_file(io.BytesIO(webm_bytes), format="webm")
        audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        return audio.raw_data  # P

    def lang_to_full(self, code: str) -> str:
        """Map a language code to its full name."""
        lang_map = {
            "en": "English",
            "es": "Spanish",
            "fr": "French",
            "zh": "Chinese"
        }
        return lang_map.get(code, code)

    def tokenize_for_tts(self, text: str, max_words: int = 3):
        """
        Split text into small chunks for streaming TTS.
        Each chunk is up to max_words, split on punctuation for natural breaks.
        """
        pattern = r"\w+|[^\w\s]"
        words = re.findall(pattern, text)
        tokens = []
        buf = []

        for w in words:
            buf.append(w)
            if len(buf) >= max_words or w in {".", "!", "?", ","}:
                tokens.append(" ".join(buf))
                buf = []

        if buf:
            tokens.append(" ".join(buf))

        return tokens


if __name__ == "__main__":
    # Initialize manager (preloads all voices)
    import asyncio

    tts = TTSManager()
    asyncio.run(tts.warmup())

    # Test Chinese TTS - Native characters
    print("🧪 Testing Chinese TTS...")
    chinese_text = "你好世界，我喜欢学习中文"  # "Hello world, I like learning Chinese"
    chinese_audio_b64 = tts.text_to_speech_base64(chinese_text, "zh")

    # NEW: Test Chinese TTS - Pinyin
    print("\n🧪 Testing Chinese Pinyin TTS...")
    pinyin_text = "Nǐ hǎo shìjiè, wǒ xǐhuān xuéxí Zhōngwén"
    pinyin_audio_b64 = tts.text_to_speech_base64(pinyin_text, "zh")

    # Save both to files
    import base64

    with open("../audio/test_chinese.wav", "wb") as f:
        f.write(base64.b64decode(chinese_audio_b64))
    with open("../audio/test_chinese_pinyin.wav", "wb") as f:
        f.write(base64.b64decode(pinyin_audio_b64))

    print("✅ Chinese TTS tests passed!")
    print(f"📁 Native: test_chinese.wav ({len(chinese_audio_b64) / 1000:.1f}KB)")
    print(f"📁 Pinyin: test_chinese_pinyin.wav ({len(pinyin_audio_b64) / 1000:.1f}KB)")

    print("🧪 Testing Whisper on Chinese TTS output...")

    # Native Chinese
    native_result = tts.whisper_model.transcribe("test_chinese.wav", language="zh")
    print(f"Native Chinese → '{native_result['text']}'")

    # Pinyin Chinese
    pinyin_result = tts.whisper_model.transcribe("test_chinese_pinyin.wav", language="zh")
    print(f"Pinyin Chinese → '{pinyin_result['text']}'")

    # Also test English model on Chinese (your code-switching case)
    print("\n🧪 English model on Chinese audio...")
    en_model_result = tts.whisper_model.transcribe("test_chinese.wav", language=None)
    print(f"medium on Chinese → '{en_model_result['text']}'")

    # Test ALL languages
    test_texts = {
        "en": "Hello, this is English!",
        "es": "¡Hola, esto es español!",
        "fr": "Bonjour, ceci est français!",
        "zh": "你好，这是中文测试！"
    }

    for lang, text in test_texts.items():
        audio_b64 = tts.text_to_speech_base64(text, lang)
        print(f"✅ {lang}: {len(audio_b64) / 1000:.1f}KB")
