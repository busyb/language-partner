from __future__ import annotations

import base64
import json
import os
import re
import tempfile
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks, status, UploadFile, File, \
    Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from llama_cpp import Llama
from starlette.responses import FileResponse

from utility.prompt_manager import PromptManager, ConversationState
from utility.UniversalTranslator import UniversalTranslator
from utility.code_switch_detector import CodeSwitchDetector
from utility.dto import TranslateResponse, TranslateRequest
from utility.llm_streamer import LLMStreamer
from utility.pinyin import PinyinGenerator
from utility.pronouce_dto import GenerateSentenceRequest, SynthesizeRequest
from utility.pronunciation_grader import PronunciationGrader
from utility.speaking_report import SpeakingReportPromptBuilder
from utility.tts_manager import TTSManager
from utility.session_manager import SessionManager

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Absolute path relative to this file
static_path = Path(__file__).parent.parent / "static"
static_path.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=static_path), name="static")


# -----------------------------
# Startup
# -----------------------------
@app.on_event("startup")
async def startup():
    # SHARED (non-session) resources
    app.state.detector = CodeSwitchDetector()
    app.state.tts_manager = TTSManager()
    app.state.pinyin = PinyinGenerator()
    app.state.silence_pcm_b64 = generate_silence_pcm()
    app.state.report_prompt_builder = SpeakingReportPromptBuilder()

    # Initialize LLM
    llm = Llama.from_pretrained(
        repo_id="Qwen/Qwen2.5-3B-Instruct-GGUF",
        filename="qwen2.5-3b-instruct-q4_k_m.gguf",
        n_ctx=2048,
        n_gpu_layers=-1,
        n_threads=8
    )
    app.state.llm_streamer = LLMStreamer(llm=llm, llm_type="llama_cpp")
    app.state.llm_streamer.warm_up()

    # NEW: Session manager
    app.state.session_manager = SessionManager(ttl_hours=2, cleanup_interval=300)

    print("✅ Server started with session management enabled")


# -----------------------------
# Helpers
# -----------------------------
def parse_lang_pair(lang_pair: str) -> tuple[str, str]:
    """Parse 'en-es' → ('en', 'es')"""
    native_lang, practicing_lang = lang_pair.split("-")
    return native_lang, practicing_lang


def generate_silence_pcm(duration_ms=20, framerate=22050):
    import numpy as np
    import io
    import wave
    import base64

    n_samples = int(framerate * duration_ms / 1000)
    silence = np.zeros(n_samples, dtype=np.int16)

    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(framerate)
        wav_file.writeframes(silence.tobytes())

    wav_buffer.seek(0)
    return base64.b64encode(wav_buffer.read()).decode()


def is_emoji_or_symbol(text):
    """Check if text is only emojis or symbols"""
    text_without_emoji = re.sub(r'[^\w\s]', '', text)
    return len(text_without_emoji.strip()) == 0


# -----------------------------
# HTTP Endpoints
# -----------------------------
@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    html_path = Path(__file__).parent.parent / "static" / "stream.html"
    with html_path.open("r", encoding="utf-8") as f:
        return f.read()


@app.get("/pronunciation", response_class=HTMLResponse)
async def serve_pronounce():
    html_path = Path(__file__).parent.parent / "static" / "pronunciation.html"
    with html_path.open("r", encoding="utf-8") as f:
        return f.read()


# -----------------------------
# WebSocket with Session Management
# -----------------------------
@app.websocket("/ws/chattime")
async def chat_ws_time_fixed(websocket: WebSocket):
    await websocket.accept()

    session_id: str | None = None
    session = None

    # Shared resources
    tts_manager = app.state.tts_manager
    detector = app.state.detector
    pinyin = app.state.pinyin
    llm_stream = app.state.llm_streamer
    session_manager = app.state.session_manager

    try:
        while True:
            msg = await websocket.receive_text()
            data = json.loads(msg)

            # ────────────── STEP 0: Session + Config ──────────────
            if data["type"] == "config":
                # Extract session_id from client
                session_id = data.get("session_id")
                if not session_id:
                    await websocket.send_json({
                        "type": "error",
                        "message": "session_id is required"
                    })
                    continue

                # Get or create session
                session = session_manager.get_or_create_session(session_id)

                lang_pair = data["lang_pair"]
                new_difficulty = data["difficulty"]

                native_lang, practicing_lang = parse_lang_pair(lang_pair)

                # Initialize conversation state if needed
                if session.conversation_state is None:
                    session.conversation_state = PromptManager.initialize_conversation(lang_pair, new_difficulty)
                    print(f"[{session_id}] Initialized new conversation state")

                # Check for language change
                if lang_pair != session.conversation_state.current_lang_pair:
                    session.counter = 0
                    print(f"[{session_id}] Reset context for new language: {lang_pair}")
                    PromptManager.reset_for_language(session.conversation_state, lang_pair, new_difficulty)
                    await websocket.send_json({"type": "status", "step": "config_updated"})

                # Check for difficulty change
                elif new_difficulty != session.conversation_state.difficulty:
                    print(f"[{session_id}] Difficulty changed to {new_difficulty}")
                    PromptManager.update_prompt_for_difficulty(session.conversation_state, new_difficulty, lang_pair)
                    await websocket.send_json({"type": "status", "step": "difficulty_updated"})

                await websocket.send_json({"type": "status", "step": "config_loaded"})

            # ────────────── STEP 1: Audio chunks ──────────────
            elif data["type"] == "audio":
                if not session or not session.conversation_state:
                    await websocket.send_json({"type": "error", "message": "No active session"})
                    continue

                native_lang, practicing_lang = parse_lang_pair(session.conversation_state.current_lang_pair)
                pcm_chunk = None

                # Raw PCM streaming
                if "pcm16_b64" in data:
                    pcm_chunk = base64.b64decode(data["pcm16_b64"])
                # Legacy WebM
                elif "webm_b64" in data:
                    webm_bytes = base64.b64decode(data["webm_b64"])
                    pcm_chunk = tts_manager.feed_webm_chunk(webm_bytes)

                if pcm_chunk:
                    partial = tts_manager.feed_pcm_to_whisper(pcm_chunk, practicing_lang)
                    if partial:
                        await websocket.send_json({"type": "partial", "text": partial})
                        session.full_transcript += partial

            # ────────────── STEP 2: End of audio → response ──────────────
            elif data["type"] == "end_audio":
                if not session or not session.conversation_state:
                    await websocket.send_json({"type": "error", "message": "No active session"})
                    continue

                conv_state = session.conversation_state
                native_lang, practicing_lang = parse_lang_pair(conv_state.current_lang_pair)
                t0 = time.perf_counter()

                # ASR finalize
                t_asr_start = time.perf_counter()
                final_text = tts_manager.feed_pcm_to_whisper(None, practicing_lang)
                if final_text.strip():
                    session.full_transcript += final_text
                    await websocket.send_json({"type": "transcript", "text": session.full_transcript})
                t_asr = time.perf_counter() - t_asr_start

                # Translation detection
                t_detect_start = time.perf_counter()
                translations = detector.detect_foreign_words(
                    session.full_transcript, practicing_lang, native_lang
                )
                t_detect = time.perf_counter() - t_detect_start
                await websocket.send_json({"type": "translations", "data": {"translations": translations}})

                # Prompt build using stateless PromptManager
                t_prompt_start = time.perf_counter()
                messages = PromptManager.build_chat_messages(conv_state, session.full_transcript)
                t_prompt = time.perf_counter() - t_prompt_start

                # LLM streaming
                t_llm_start = time.perf_counter()
                assistant_text = ""
                AUDIO_WINDOW_MS = 300
                first_audio_sent = False
                first_token_time: float | None = None
                second_token_time: float | None = None
                llm_start_time = time.perf_counter()
                token_buffer = ""
                print(json.dumps(messages, indent=2, ensure_ascii=False))

                for chunk in llm_stream.stream(messages):
                    token = chunk
                    if not token.strip():
                        continue

                    assistant_text += token
                    token_buffer += token

                    # Send pre-buffered silence
                    if not first_audio_sent:
                        await websocket.send_json({
                            "type": "audio_chunk",
                            "pcm16_b64": app.state.silence_pcm_b64,
                            "meta": "prebuffer"
                        })
                        first_audio_sent = True

                    now = time.perf_counter()
                    if first_token_time is None:
                        first_token_time = now - llm_start_time
                    elif second_token_time is None:
                        second_token_time = now - llm_start_time

                    # Phrase-buffered TTS
                    phrase_end = re.search(r"[.?!,;。！？、]|\n", token_buffer)
                    should_flush = (
                            phrase_end or
                            (len(token_buffer.strip()) >= 12 if practicing_lang == "zh" else len(
                                token_buffer.split()) >= 10)
                    )

                    if should_flush:
                        phrase_text = token_buffer.strip()
                        token_buffer = ""

                        if not phrase_text:
                            continue

                        pcm_window = b""
                        window_frames = 0
                        framerate = 24000
                        frames_per_window = int(framerate * (AUDIO_WINDOW_MS / 1000))

                        for pcm_b64 in tts_manager.text_to_pcm_chunks_streaming(phrase_text, practicing_lang):
                            pcm_bytes = base64.b64decode(pcm_b64)
                            pcm_window += pcm_bytes
                            window_frames += len(pcm_bytes) // 2

                            if window_frames >= frames_per_window:
                                await websocket.send_json({
                                    "type": "audio_chunk",
                                    "pcm16_b64": base64.b64encode(pcm_window).decode()
                                })
                                pcm_window = b""
                                window_frames = 0

                        if pcm_window:
                            await websocket.send_json({
                                "type": "audio_chunk",
                                "pcm16_b64": base64.b64encode(pcm_window).decode()
                            })

                total_llm_time = time.perf_counter() - llm_start_time

                # Flush remaining tokens
                if token_buffer.strip() and not is_emoji_or_symbol(token_buffer.strip()):
                    phrase_text = token_buffer.strip()
                    pcm_window = b""
                    window_frames = 0
                    framerate = 24000
                    frames_per_window = int(framerate * (AUDIO_WINDOW_MS / 1000))

                    for pcm_b64 in tts_manager.text_to_pcm_chunks_streaming(phrase_text, practicing_lang):
                        pcm_bytes = base64.b64decode(pcm_b64)
                        pcm_window += pcm_bytes
                        window_frames += len(pcm_bytes) // 2

                        if window_frames >= frames_per_window:
                            await websocket.send_json({
                                "type": "audio_chunk",
                                "pcm16_b64": base64.b64encode(pcm_window).decode()
                            })
                            pcm_window = b""
                            window_frames = 0

                    if pcm_window:
                        await websocket.send_json({
                            "type": "audio_chunk",
                            "pcm16_b64": base64.b64encode(pcm_window).decode()
                        })

                print(f"\n⏱️ [{session_id}] WS CHAT TIMINGS")
                print(f"  ASR finalize: {t_asr:.3f}s")
                print(f"  Translation detect: {t_detect:.3f}s")
                print(f"  Prompt build: {t_prompt:.3f}s")
                print(f"  First token latency: {first_token_time:.3f}s")
                print(f"  Total LLM streaming: {total_llm_time:.3f}s")

                # Send final results
                await websocket.send_json({"type": "assistant_text", "text": assistant_text})

                assistant_segments = (
                    pinyin.add_pinyin_to_text_structured(assistant_text) if practicing_lang == "zh" else []
                )

                # Update conversation history using stateless methods
                PromptManager.update_history_user(conv_state, session.full_transcript)
                PromptManager.update_history_assistant(conv_state, assistant_text)

                await websocket.send_json({
                    "type": "assistant_segments",
                    "assistant_segments": assistant_segments
                })

                session.counter += 1
                await websocket.send_json({"type": "user_response_counter", "text": session.counter})

                session.full_transcript = ""
                await websocket.send_json({"type": "done"})

    except WebSocketDisconnect:
        if session_id:
            print(f"[{session_id}] Client disconnected")
        else:
            print("Client disconnected")


# -----------------------------
# Pronunciation Endpoints (Session-aware)
# -----------------------------
@app.post("/score-pronunciation")
async def score_pronunciation(
        audio: UploadFile = File(...),
        expected_text: str = Form(...),
        language: str = Form(...),
        session_id: str = Form(...)
):
    """Score pronunciation with session management"""
    session_manager = app.state.session_manager
    session = session_manager.get_or_create_session(session_id)

    # Check if language changed
    if session.pronounce_lang != language:
        session.pronounce_counter = 0
        session.pronounce_lang = language
        session.pronunciation_history = []

    session.pronounce_counter += 1

    # Create a PronunciationGrader instance per session if needed
    grader = PronunciationGrader()

    result = app.state.tts_manager.transcribe_audio(audio, language)
    heard_text = result.get("text", "").strip()

    scores = grader.grade_words(
        segments=result["segments"],
        expected_text=expected_text,
        language=language,
        heard=heard_text,
        counter=session.pronounce_counter
    )

    # Store in session
    session.pronunciation_history.append({
        "expected": expected_text,
        "heard": heard_text,
        "scores": scores,
        "counter": session.pronounce_counter
    })

    return scores


@app.post("/report/pronunciation")
async def generate_pronunciation_report(request: dict):
    """Generate pronunciation report from session data"""
    session_id = request.get("session_id")
    if not session_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="session_id is required"
        )

    session_manager = app.state.session_manager
    session = session_manager.get_or_create_session(session_id)

    if not session.pronunciation_history:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No pronunciation data available for this session"
        )

    builder = app.state.report_prompt_builder
    prompt = builder.build_pronunciation_drill_report_prompt(
        request.get("language"),
        session.pronunciation_history
    )

    try:
        llm_text = app.state.llm_streamer.call_llm_not_stream(prompt)

        if not llm_text:
            return {}

        if llm_text.startswith("```") and llm_text.endswith("```"):
            llm_text = "\n".join(llm_text.split("\n")[1:-1])

        llm_text = llm_text.strip().strip('"').strip("'")

        try:
            return json.loads(llm_text)
        except json.JSONDecodeError:
            try:
                fixed_text = llm_text.replace("'", '"')
                return json.loads(fixed_text)
            except json.JSONDecodeError:
                return {}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate report: {str(e)}"
        )


@app.post("/report/conversation")
async def generate_conversation_report(request: dict):
    """Generate conversation report from session data"""
    session_id = request.get("session_id")
    if not session_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="session_id is required"
        )

    session_manager = app.state.session_manager
    session = session_manager.get_or_create_session(session_id)

    if not session.conversation_state:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No conversation data available for this session"
        )

    # Get conversation text using stateless method
    conversation = PromptManager.get_conversation_text(session.conversation_state)

    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No conversation messages available"
        )

    lang_pair = request.get("lang_pair")
    difficulty = request.get("difficulty")
    split = lang_pair.split("-")

    builder = app.state.report_prompt_builder
    prompt = builder.build_conversation_report_prompt(
        split[1],
        difficulty,
        conversation
    )

    try:
        llm_text = app.state.llm_streamer.call_llm_not_stream(prompt)

        if not llm_text:
            return {}

        if llm_text.startswith("```") and llm_text.endswith("```"):
            llm_text = "\n".join(llm_text.split("\n")[1:-1])

        llm_text = llm_text.strip().strip('"').strip("'")

        try:
            return json.loads(llm_text)
        except json.JSONDecodeError:
            try:
                fixed_text = llm_text.replace("'", '"')
                return json.loads(fixed_text)
            except json.JSONDecodeError:
                return {}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate report: {str(e)}"
        )


# -----------------------------
# Other Endpoints
# -----------------------------
@app.post("/translate", response_model=TranslateResponse)
def translate(req: TranslateRequest):
    start = time.perf_counter()
    try:
        translator = UniversalTranslator()
        translation = translator.translate_single(req.text, req.from_code, req.to_code)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Translation failed: {str(e)}")

    latency_ms = (time.perf_counter() - start) * 1000
    return TranslateResponse(
        text=translation,
        text_language=req.to_code,
        latency_ms=round(latency_ms, 2)
    )


@app.post("/translate/sentence", response_model=TranslateResponse)
def translate_sentence(req: TranslateRequest):
    start = time.perf_counter()
    try:
        translator = UniversalTranslator()
        translation = translator.translate_sentence(req.text, req.from_code, req.to_code)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Translation failed: {str(e)}")

    latency_ms = (time.perf_counter() - start) * 1000
    return TranslateResponse(
        text=translation["text"],
        text_language=req.to_code,
        latency_ms=round(latency_ms, 2)
    )


@app.post("/api/generate-sentence")
async def generate_sentence(request: GenerateSentenceRequest):
    lang = request.language
    sentence = app.state.llm_streamer.generate_sentence_file(lang)
    segments = (
        app.state.pinyin.add_pinyin_to_text_structured(sentence) if lang == "zh" else []
    )
    return {"sentence": sentence, "segments": segments}


@app.post("/api/synthesize-sentence")
async def synthesize_sentence(request: SynthesizeRequest):
    b64_audio = app.state.tts_manager.text_to_speech_base64(request.text, request.language)
    raw_bytes = base64.b64decode(b64_audio)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_file.write(raw_bytes)
        temp_file.flush()
        temp_path = temp_file.name

    background_tasks = BackgroundTasks()
    background_tasks.add_task(lambda: os.remove(temp_path))

    return FileResponse(
        temp_path,
        media_type="audio/wav",
        filename="sentence.wav",
        background=background_tasks
    )
@app.post("/test/conversation")
async def generate_sentence(request: GenerateSentenceRequest):
    lang = request.language
    sentence = app.state.llm_streamer.generate_sentence_file(lang)
    segments = (
        app.state.pinyin.add_pinyin_to_text_structured(sentence) if lang == "zh" else []
    )
    return {"sentence": sentence, "segments": segments}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)