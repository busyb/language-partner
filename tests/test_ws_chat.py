import base64
import json
from pathlib import Path
import asyncio
import base64
import json
import time
from pathlib import Path
import io
from typing import Iterable

import numpy as np
import piper
import uuid
from fastapi.testclient import TestClient
# from app.main import app  # adjust if your main.py is elsewhere
from tempfile import NamedTemporaryFile
import subprocess

from fastapi.testclient import TestClient
from llama_cpp import Llama
from starlette.websockets import WebSocketDisconnect

from app.main import app
import pytest
from unittest.mock import MagicMock

from utility.llm_streamer import LLMStreamer
from utility.prompt_manager import PromptManager

client = TestClient(app)


@pytest.fixture(autouse=True)
def mock_pm_tts(monkeypatch):
    app.state.pm = MagicMock()
    app.state.tts_manager = MagicMock()
    app.state.detector = MagicMock()
    app.state.pinyin = MagicMock()
    app.state.stream_model = MagicMock()
    app.state.current_lang_pair = "en-es"


def test_ws_connects():
    with client.websocket_connect("/ws/chat") as ws:
        ws.send_text(json.dumps({"type": "config", "lang_pair": "en-es"}))
        try:
            msg = ws.receive_json()
            assert msg["type"] == "status"
        except WebSocketDisconnect:
            # Normal closure, can ignore in test
            pass


@pytest.mark.asyncio
async def test_ws_chat_stream():
    """
    Full async WebSocket test for chat streaming:
    - Streams WAV as WebM chunks
    - Tests LLM, TTS, transcript, translations, and assistant text
    """
    AUDIO_DIR = Path(__file__).parent / "audio"
    AUDIO_FILES = ["english.wav"]  # Replace with your test files
    LANG_PAIR = "en-es"
    CHUNK_DELAY = 0.05  # seconds between audio chunks

    # Use TestClient for FastAPI WebSocket
    with TestClient(app) as client:
        try:
            with client.websocket_connect("/ws/chat") as websocket:

                # STEP 0: Send config
                websocket.send_text(json.dumps({"type": "config", "lang_pair": LANG_PAIR}))
                msg = websocket.receive_json()
                print("Config ack:", msg)

                # STEP 1: Stream audio files as WebM chunks
                for audio_file in AUDIO_FILES:
                    path = AUDIO_DIR / audio_file
                    webm_bytes = wav_to_webm_bytes(path)
                    webm_b64 = base64.b64encode(webm_bytes).decode()

                    websocket.send_text(json.dumps({
                        "type": "audio",
                        "webm_b64": webm_b64
                    }))
                    await asyncio.sleep(CHUNK_DELAY)

                # STEP 2: End of audio
                websocket.send_text(json.dumps({"type": "end_audio"}))

                first_audio_latency = None
                assistant_text_received = False
                transcript = ""
                translations = []

                # Receive WebSocket messages
                while True:
                    try:
                        msg = websocket.receive_json()
                    except WebSocketDisconnect:
                        print("Client disconnected (expected at end)")
                        break

                    if msg["type"] == "transcript":
                        transcript = msg["text"]
                        print("transcript:", transcript)

                    elif msg["type"] == "translations":
                        translations = msg["data"]["translations"]
                        print("Translations received:", translations)

                    elif msg["type"] == "audio_chunk" and first_audio_latency is None:
                        first_audio_latency = 0  # Could measure time if needed
                        print("Audio chunk received, size:", len(base64.b64decode(msg["pcm16_b64"])))

                    elif msg["type"] == "assistant_text":
                        assistant_text_received = True
                        print("Assistant text:", msg["text"])

                    elif msg["type"] == "assistant_segments":
                        print("Assistant segments:", msg["assistant_segments"])

                    elif msg["type"] == "done":
                        print("Conversation done")
                        break

                # Assertions
                assert assistant_text_received, "No assistant text received"
                assert transcript, "No transcript received"
                assert translations, "No translations received"

        finally:
            try:
                websocket.close()
            except Exception:
                pass


@pytest.mark.asyncio
async def test_ws_chat_back_and_forth():
    """
    Test 5-round conversation to see if LLM speeds up after warm-up.
    """
    AUDIO_DIR = Path(__file__).parent / "audio"
    AUDIO_FILE = "english.wav"  # Replace with a real test file
    LANG_PAIR = "en-es"
    CHUNK_DELAY = 0.05

    # Measure per-round first-token latency
    round_latencies = []

    with TestClient(app) as client:
        with client.websocket_connect("/ws/chattime") as websocket:
            # Send initial config
            websocket.send_text(json.dumps({"type": "config", "lang_pair": LANG_PAIR}))
            msg = websocket.receive_json()
            print("Config ack:", msg)

            for round_num in range(1, 2):
                print(f"\n=== Round {round_num} ===")

                # Convert WAV to WebM
                path = AUDIO_DIR / AUDIO_FILE
                webm_bytes = wav_to_webm_bytes(path)
                webm_b64 = base64.b64encode(webm_bytes).decode()

                # Send audio
                websocket.send_text(json.dumps({"type": "audio", "webm_b64": webm_b64}))
                await asyncio.sleep(CHUNK_DELAY)

                # End audio
                websocket.send_text(json.dumps({"type": "end_audio"}))

                # Wait for assistant response and measure first audio chunk latency
                first_audio_time = None
                start_time = time.perf_counter()
                while True:
                    try:
                        msg = websocket.receive_json()
                    except WebSocketDisconnect:
                        print("Client disconnected")
                        break

                    if msg["type"] == "audio_chunk" and first_audio_time is None:
                        first_audio_time = time.perf_counter() - start_time
                        print(f"First audio chunk latency: {first_audio_time:.3f}s")

                    elif msg["type"] == "assistant_text":
                        print("Assistant text:", msg["text"])

                    elif msg["type"] == "done":
                        break

                if first_audio_time is not None:
                    round_latencies.append(first_audio_time)

    print("\n=== Round latencies (seconds) ===")
    for i, latency in enumerate(round_latencies, 1):
        print(f"Round {i}: {latency:.3f}s")


def wav_to_webm_bytes(wav_path: Path) -> bytes:
    """
    Convert a WAV file to WebM Opus in memory for testing.
    """
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".webm") as tmp_webm:
        subprocess.run([
            "ffmpeg",
            "-y",
            "-i", str(wav_path),
            "-c:a", "libopus",
            "-b:a", "64k",
            "-f", "webm",
            tmp_webm.name
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        tmp_webm.seek(0)
        return tmp_webm.read()

@pytest.mark.asyncio
async def test_ws_chat_back_and_forth_playable():
    """
    Test 5-round conversation and produce a playable WAV from streamed TTS audio.
    """
    AUDIO_DIR = Path(__file__).parent / "audio"
    AUDIO_FILE = "english.wav"  # Replace with a real test file
    LANG_PAIR = "en-es"
    CHUNK_DELAY = 0.05

    round_latencies = []

    with TestClient(app) as client:
        with client.websocket_connect("/ws/chattime") as websocket:
            websocket.send_text(json.dumps({"type": "config", "lang_pair": LANG_PAIR}))
            msg = websocket.receive_json()
            print("Config ack:", msg)

            for round_num in range(1, 6):  #how many times to loop
                print(f"\n=== Round {round_num} ===")

                path = AUDIO_DIR / AUDIO_FILE
                webm_bytes = wav_to_webm_bytes(path)
                webm_b64 = base64.b64encode(webm_bytes).decode()

                websocket.send_text(json.dumps({"type": "audio", "webm_b64": webm_b64}))
                await asyncio.sleep(CHUNK_DELAY)
                websocket.send_text(json.dumps({"type": "end_audio"}))

                first_audio_time = None
                start_time = time.perf_counter()
                pcm_chunks = []  # <- store all PCM16 frames

                while True:
                    try:
                        msg = websocket.receive_json()
                    except WebSocketDisconnect:
                        print("Client disconnected")
                        break

                    if msg["type"] == "audio_chunk":
                        pcm_chunks.append(base64.b64decode(msg["pcm16_b64"]))
                        if first_audio_time is None:
                            first_audio_time = time.perf_counter() - start_time
                            print(f"First audio chunk latency: {first_audio_time:.3f}s")

                    elif msg["type"] == "assistant_text":
                        print("Assistant text:", msg["text"])

                    elif msg["type"] == "done":
                        break

                # Write WAV for this round
                output_file = f"round_{round_num}_output.wav"
                with wave.open(output_file, "wb") as wf:
                    wf.setnchannels(1)  # mono
                    wf.setsampwidth(2)  # PCM16 = 2 bytes
                    wf.setframerate(22050)  # adjust to your TTS voice sample rate
                    for chunk in pcm_chunks:
                        wf.writeframes(chunk)
                print(f"Saved WAV: {output_file}")

                # Optional: play audio automatically
                wave_obj = sa.WaveObject.from_wave_file(output_file)
                play_obj = wave_obj.play()
                play_obj.wait_done()

                if first_audio_time is not None:
                    round_latencies.append(first_audio_time)

    print("\n=== Round latencies (seconds) ===")
    for i, latency in enumerate(round_latencies, 1):
        print(f"Round {i}: {latency:.3f}s")


import asyncio
import base64
import json
import time
import wave
from pathlib import Path

import pytest
import simpleaudio as sa
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.main import app  # your FastAPI app

import asyncio
import base64
import json
import wave
from pathlib import Path

import pytest
import simpleaudio as sa
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.main import app
# Import Piper TTS engine (adjust import to your actual Piper wrapper)
from piper import PiperVoice

BASE_DIR = Path(__file__).parent.parent  # adjust as needed

# Preset user phrases
USER_PHRASES = {
    "en": [
        "Good morning! How are you?",
        "I would like to practice English with you.",
        "Can we talk about hobbies?",
        "What do you like to do in your free time?",
        "Thank you, that was helpful!"
    ],
    "fr": [
        "Bonjour! Comment ça va?",
        "Je voudrais pratiquer le français avec vous.",
        "Pouvons-nous parler des loisirs?",
        "Qu'est-ce que vous aimez faire pendant votre temps libre?",
        "Merci, c'était utile!"
    ],
    "es": [
        "¡Buenos días! ¿Cómo estás?",
        "Me gustaría practicar español contigo.",
        "¿Podemos hablar sobre pasatiempos?",
        "¿Qué te gusta hacer en tu tiempo libre?",
        "¡Gracias, eso fue útil!"
    ],
    "zh": [
        "早上好！你好吗？",
        "我想和你练习中文。",
        "我们可以聊聊爱好吗？",
        "你空闲时间喜欢做什么？",
        "谢谢，这很有帮助！"
    ]
}

# Piper voice models
VOICE_PATHS = {
    "en": BASE_DIR / "audio" / "piper_models" / "en_US-amy-medium.onnx",
    "fr": BASE_DIR / "audio" / "piper_models" / "fr_FR-siwis-medium.onnx",
    "es": BASE_DIR / "audio" / "piper_models" / "es_ES-sharvard-medium.onnx",
    "zh": BASE_DIR / "audio" / "piper_models" / "zh_CN-huayan-medium.onnx",
}

import numpy as np
import wave
from pathlib import Path

import numpy as np
import wave
from pathlib import Path


def read_pcm16_from_wav(wav_path: Path):
    with wave.open(str(wav_path), "rb") as wf:
        assert wf.getsampwidth() == 2, "Expected PCM16 WAV"
        assert wf.getnchannels() == 1, "Expected mono WAV"
        pcm_bytes = wf.readframes(wf.getnframes())
    return pcm_bytes


# Split PCM bytes into chunks
def chunk_bytes(pcm_bytes: bytes, chunk_size: int = 4096):
    for i in range(0, len(pcm_bytes), chunk_size):
        yield pcm_bytes[i:i + chunk_size]


lang_pairs = ["en-es"] # ["en-es", "en-fr", "en-zh"] #["en-zh"] #


@pytest.mark.asyncio
async def test_ws_chat_user_practice_piper(play_sound=False):
    AUDIO_DIR = Path(__file__).parent / "audio"
    diffToSend = "B1"
    import uuid
    uuid = uuid_str = str(uuid.uuid4())

    with TestClient(app) as client:
        for LANG_PAIR in lang_pairs:
            native_lang, practicing_lang = LANG_PAIR.split("-")
            print(f"\n=== Testing language pair: {LANG_PAIR} ===")

            with client.websocket_connect("/ws/chattime") as websocket:
                # Send config
                websocket.send_text(json.dumps({"type": "config", "session_id": uuid, "lang_pair": LANG_PAIR, "difficulty": diffToSend}))
                msg = websocket.receive_json()
                print("Config ack:", msg)

                if LANG_PAIR == "en-es":
                    # Send English then Spanish audio for the same pair
                    user_langs = ["es"]
                else:
                    # Only send the practicing language once
                    user_langs = [practicing_lang]

                for lang_code in user_langs:
                    for round_num in range(1, 6):
                        print(f"\n--- Round {round_num}")

                        user_wav_path = AUDIO_DIR / f"round_{round_num}_output_{lang_code}.wav"
                        assert user_wav_path.exists(), user_wav_path

                        pcm_bytes = read_pcm16_from_wav(user_wav_path)

                        for i, chunk in enumerate(chunk_bytes(pcm_bytes, chunk_size=4096)):
                            websocket.send_text(json.dumps({
                                "type": "audio",
                                "pcm16_b64": base64.b64encode(chunk).decode()
                            }))
                            await asyncio.sleep(0.02)  # mimic browser streaming

                        websocket.send_text(json.dumps({"type": "end_audio"}))

                        # Collect AI response audio
                        pcm_chunks = []
                        start_time = asyncio.get_event_loop().time()
                        first_audio_time = None
                        while True:
                            try:
                                msg = websocket.receive_json()
                            except WebSocketDisconnect:
                                print("Client disconnected")
                                break

                            if msg["type"] == "audio_chunk":
                                pcm_chunks.append(base64.b64decode(msg["pcm16_b64"]))
                                if first_audio_time is None:
                                    first_audio_time = asyncio.get_event_loop().time() - start_time
                                    print(f"First AI audio chunk latency: {first_audio_time:.3f}s")
                            elif msg["type"] == "translations":
                                print("Captured translations:", msg["data"]["translations"])
                            elif msg["type"] == "assistant_text":
                                print("AI response text:", msg["text"])
                            elif msg["type"] == "done":
                                break

                        # Save AI response WAV
                        if play_sound:
                            ai_wav_path = AUDIO_DIR / "response" / f"round_{round_num}_ai_{lang_code}.wav"
                            with wave.open(str(ai_wav_path), "wb") as wf:
                                wf.setnchannels(1)
                                wf.setsampwidth(2)
                                wf.setframerate(22050)
                                for chunk in pcm_chunks:
                                    wf.writeframes(chunk)
                            # print(f"Saved AI response WAV: {ai_wav_path}")

                            # Optionally play AI response
                            wave_obj = sa.WaveObject.from_wave_file(str(ai_wav_path))
                            play_obj = wave_obj.play()
                            play_obj.wait_done()


def text_to_wav_piper(text: str, lang_code: str, output_path: Path):
    """
    Generate WAV from Piper TTS, handling generator output robustly.
    Converts AudioChunk objects to PCM16 bytes for wave.writeframes().
    """
    tts = PiperVoice.load(model_path=str(VOICE_PATHS[lang_code]))
    audio_gen = tts.synthesize_wav(text)  # generator

    pcm_arrays = []

    for chunk in audio_gen:
        # Only handle AudioChunk objects or numpy arrays
        if hasattr(chunk, "data"):
            arr = np.array(chunk.data, dtype=np.int16)
        elif isinstance(chunk, np.ndarray):
            arr = chunk.astype(np.int16)
        else:
            # Skip anything else
            continue

        if arr.size > 0:
            pcm_arrays.append(arr.flatten())

    # If nothing was generated, produce 0.1s silence
    if len(pcm_arrays) == 0:
        pcm16 = np.zeros(2205, dtype=np.int16)  # 22050Hz * 0.1s
    else:
        pcm16 = np.concatenate(pcm_arrays)

    # Convert to bytes and write WAV
    audio_bytes = pcm16.tobytes()
    with wave.open(str(output_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(22050)
        wf.writeframes(audio_bytes)


def generate_all_user_audio(user_phrases: dict[str, list[str]]):
    """
    Generate all user audio WAVs ahead of time.
    Files will be named: round_<n>_output_<lang>.wav
    """
    AUDIO_DIR = Path(__file__).parent / "audio/"

    for lang_code, phrases in user_phrases.items():
        for round_num, text in enumerate(phrases, start=1):
            output_file = AUDIO_DIR / f"round_{round_num}_output_{lang_code}.wav"
            print(f"Generating {output_file} ...")
            voice = PiperVoice.load(model_path=str(VOICE_PATHS[lang_code]))
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)  # PCM16
                wav_file.setframerate(24000)  # match Piper voice
                # Piper writes PCM16 bytes directly to wave file
                voice.synthesize_wav(text, wav_file)

            # Write buffer to disk
            wav_buffer.seek(0)
            with open(output_file, "wb") as f:
                f.write(wav_buffer.read())


def test_conversation():
    # --- Init LLM ---
    base_llm = Llama.from_pretrained(
        repo_id="Qwen/Qwen2.5-3B-Instruct-GGUF",
        filename="qwen2.5-3b-instruct-q4_k_m.gguf",
        n_ctx=2048,
        n_gpu_layers=-1,
        n_threads=8
    )
    llm = LLMStreamer(llm=base_llm, llm_type="llama_cpp")

    # --- Init conversation ---
    manager = PromptManager()
    state = manager.initialize_conversation(
        lang_pair="en-es",
        difficulty="B1"
    )

    # --- User starts first ---
    user_response = " Me gustaría practicar conversación en español."

    for turn in range(5):
        print(f"\n--- Turn {turn + 1} ---")
        print("User:", user_response)

        # 1️⃣ Store user turn

        # 2️⃣ Build full chat context from history
        messages = manager.build_chat_messages(state=state,user_input=user_response)

        manager.update_history_user(state, user_response)

        print(json.dumps(messages, indent=2, ensure_ascii=False))

        # 3️⃣ Assistant responds using full context
        ai_response = llm.call_llm_not_stream(messages)

        # 4️⃣ Store assistant turn
        manager.update_history_assistant(state, ai_response)

        # 5️⃣ Simulated learner generates next response
        messages = build_test_user_messages("es", "B1", ai_response)
        user_response = llm.respondToAI(messages)
        print("AI:", ai_response)

    # --- Assertions ---
    assert len(state.messages) > 5

def build_test_user_messages(lang: str, difficulty: str, ai_response: str) -> list[dict]:
    """
    Build messages for test user responding to AI.
    Optimized for Qwen2.5-3B - minimal and direct.

    Args:
        lang_pair: Language pair (e.g., "en-es")
        difficulty: CEFR level (e.g., "B1")
        ai_response: The AI's message to respond to

    Returns:
        List of message dicts for LLM
    """

    LANG_MAP = {
        "en": "English",
        "es": "Spanish",
        "fr": "French",
        "zh": "Chinese",
    }
    target_lang = LANG_MAP.get(lang)

    # Difficulty constraints
    constraints = {
        "A1": "1-2 very simple sentences.",
        "A2": "2 simple sentences.",
        "B1": "2-3 natural sentences.",
        "B2": "3-4 conversational sentences.",
        "C1": "Natural, native-like response.",
        "C2": "Fully natural native speaker."
    }
    constraint = constraints.get(difficulty.upper(), constraints["B1"])

    # Language-specific prompts
    if lang == "es":
        system = f"""Eres Alex, un neoyorquino aprendiendo español. Nivel: {difficulty}
Estás chateando con un amigo. {constraint}

Habla de tu vida: pizza, béisbol, el metro, tu trabajo.
No ofrezcas ayuda. Solo chatea como amigo.
Responde natural, sin saludos extra."""

        user = f"Tu amigo dice: '{ai_response}'\nResponde natural en español."

    elif lang == "fr":
        system = f"""Tu es Alex, un New-Yorkais qui apprend le français. Niveau: {difficulty}
Tu discutes avec un ami. {constraint}

Parle de ta vie: pizza, baseball, métro, boulot.
N'offre pas d'aide. Discute comme un ami.
Réponds naturellement, sans salutations."""

        user = f"Ton ami dit: '{ai_response}'\nRéponds naturellement en français."

    elif lang == "zh":
        system = f"""你是Alex，一个学中文的纽约人。等级：{difficulty}
你在和朋友聊天。{constraint}

聊你的生活：披萨、棒球、地铁、工作。
不要提供帮助。像朋友一样聊天。
自然回复，不用额外打招呼。"""

        user = f"你朋友说：'{ai_response}'\n用中文自然回复。"

    else:
        system = f"""You're Alex, a New Yorker learning {target_lang}. Level: {difficulty}
You're chatting with a friend. {constraint}

Talk about your life: pizza, baseball, subway, work.
Don't offer help. Just chat as a friend.
Respond naturally, no extra greetings."""

        user = f"Your friend says: '{ai_response}'\nRespond naturally in {target_lang}."

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ]


def test_generate_all_user_audio():
    generate_all_user_audio(USER_PHRASES)
