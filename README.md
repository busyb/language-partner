# Navigate to your project root
cd /path/to/your/project

# 🌍 AI Language Learning Voice Chat

An intelligent, real-time voice conversation application for practicing foreign languages with AI. Built with FastAPI, React, and multiple AI models to provide immersive language learning through natural conversations.

## 🎯 What It Does

This application creates an AI-powered language tutor that you can speak with in real-time. It:

- **Listens** to your speech in your target language
- **Transcribes** what you said using OpenAI's Whisper
- **Detects** when you accidentally switch languages (code-switching)
- **Responds** naturally in the target language with contextual AI conversation
- **Speaks back** using realistic text-to-speech
- **Provides translations** on-demand for words and sentences
- **Adapts difficulty** to your proficiency level (A1-C2 / HSK 1-6)
- **Generates performance reports** with personalized feedback

## 🎬 Demo

![Voice Chat Interface](demo.png)

**Supported Languages:**
- 🇪🇸 Spanish ↔️ 🇺🇸 English
- 🇫🇷 French ↔️ 🇺🇸 English  
- 🇨🇳 Mandarin Chinese ↔️ 🇺🇸 English

## 🤖 AI Architecture

### Speech Recognition (STT)
- **Model**: OpenAI Whisper (medium/large)
- **Function**: Converts your spoken audio to text
- **Features**: Multi-language support, robust to accents and background noise
- **API**: `faster-whisper` for optimized inference

### Language Understanding & Response
- **Model**:  (Qwen2.5) 
- **Function**: 
  - Understands conversational context
  - Generates natural, level-appropriate responses
  - Detects code-switching (when you accidentally use English in Spanish, etc.)
  - Provides grammar corrections and vocabulary suggestions
  - Maintains conversation history for contextual awareness
- **Prompting Strategy**: 
  - System prompts tailored to proficiency level (A1-C2, HSK 1-6)
  - Context-aware conversation management
  - Educational scaffolding (hints, corrections, encouragement)

### Text-to-Speech (TTS)
- **Primary Engine**: Piper TTS (offline, fast)
- **Fallback**: System TTS
- **Function**: Converts AI responses to natural speech
- **Features**: 
  - Multiple voice models per language
  - Low latency streaming
  - Pitch-preserving speed adjustment (0.6x - 1.8x)

### Translation
- **Model**: Argos Translate (offline, privacy-friendly)
- **Function**: Provides instant word and sentence translations
- **Features**:
  - Click any word for translation
  - Cached translations for performance
  - Pronunciation playback

### Code-Switch Detection
- **Model**: Custom FastText-based language detection
- **Function**: Identifies when you accidentally switch languages
- **Highlights**: Yellow-highlighted words with automatic translations

### Chinese Language Support
- **Pinyin Generation**: `pypinyin` library
- **Function**: Shows pinyin pronunciation above Chinese characters
- **Tokenization**: Word-level segmentation for clickable vocabulary

### Performance Analysis
- **Model**: Claude with structured output prompting
- **Function**: Analyzes conversation transcripts to generate:
  - Overall proficiency scores (vocabulary, grammar, fluency, pronunciation)
  - Strengths identification
  - Key issues with examples
  - Level alignment assessment
  - Personalized action plan for improvement
- **Trigger**: After 10+ conversation exchanges

## 🏗️ System Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend (React)                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Voice Input  │  │  Real-time   │  │  Translation │      │
│  │  (WebRTC)    │  │  Transcript  │  │   Popups     │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                            │ WebSocket
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Backend                           │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │           WebSocket Manager (Async)                   │  │
│  │  • Audio streaming                                    │  │
│  │  • Real-time transcription                            │  │
│  │  • Response streaming                                 │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Whisper    │  │    Claude    │  │  Piper TTS   │     │
│  │  (Faster)    │  │  API Client  │  │   (Offline)  │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │    Argos     │  │   FastText   │  │   Pypinyin   │     │
│  │  Translate   │  │   LangDet    │  │  (Chinese)   │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

## 🎓 How AI Enables Language Learning

### 1. **Adaptive Conversation**
Claude adjusts its vocabulary, grammar complexity, and speaking speed based on your selected proficiency level:
- **A1 (Beginner)**: Simple present tense, everyday vocabulary, short sentences
- **C2 (Mastery)**: Idioms, complex grammar, nuanced expressions

### 2. **Natural Language Processing**
- **Context retention**: Remembers previous exchanges in the conversation
- **Topic continuation**: Builds on your responses naturally
- **Error correction**: Gently reformulates your mistakes without interrupting flow

### 3. **Code-Switch Detection**
Uses language detection ML to identify when you slip into your native language:
```
You: "Me gusta el libro, but it's expensive"
                          ^^^^ Detected English
AI: Shows translation + continues in Spanish
```

### 4. **Intelligent Feedback**
After analyzing your conversation, Claude generates structured feedback:
- **Quantitative scores**: Grammar, vocabulary, fluency (0-100)
- **Qualitative analysis**: Identifies patterns in your errors
- **Actionable recommendations**: Specific skills to practice next

### 5. **Multi-Modal Learning**
- **Audio**: Practice pronunciation and listening comprehension
- **Visual**: See text transcripts and translations
- **Interactive**: Click words for instant definitions and audio playback

## 🚀 Getting Started

### Prerequisites
- Docker & Docker Compose
- Tailscale account (optional, for HTTPS deployment)

### Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/ai-language-learning.git
cd ai-language-learning
```

2. **Set up environment variables**
```bash
# Create .env file
cat > .env << EOF
SESSION_TTL_HOURS=2
CLEANUP_INTERVAL_SECONDS=300
EOF
```

3. **Start the application**
```bash
docker-compose up -d
```

4. **Access the app**
- Local development: http://localhost:8001
- Production: See deployment section below

### First Time Setup

The first run will download AI models (~2-5 GB):
- Whisper models (speech recognition)
- Piper voice models (text-to-speech)
- Argos translation models
- Language detection models

**This may take 10-30 minutes depending on your internet connection.**

## 🌐 Production Deployment with HTTPS

### Option 1: Tailscale (Recommended - No Domain Needed)

1. **Install Tailscale on your server**
```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
```

2. **Start the application**
```bash
docker-compose up -d
```

3. **Enable HTTPS serving**
```bash
# Serve on your Tailscale network
sudo tailscale serve https / http://localhost:8001

# Optional: Enable public access (Funnel)
sudo tailscale funnel 443 on
```

4. **Get your HTTPS URL**
```bash
tailscale status
# Your app is now at: https://your-machine.your-tailnet.ts.net
```

### Option 2: Traditional SSL (With Domain)

See `docker-compose.yml` for Nginx + Let's Encrypt configuration.

## 🎮 Usage

1. **Select your language pair** (e.g., English → Spanish)
2. **Choose proficiency level** (A1/HSK1 → C2/HSK6)
3. **Press and hold** the microphone button (or spacebar)
4. **Speak** in your target language
5. **Release** to stop recording
6. **Listen** to AI's response
7. **Click any word** for instant translation
8. **Adjust playback speed** (0.6x - 1.8x with pitch preservation)

### Advanced Features

- **Replay responses**: Re-listen to AI's last message
- **Sentence translation**: Click the AI avatar to translate full sentences
- **Conversation report**: After 10+ exchanges, get detailed performance feedback
- **Practice mode**: Access `/pronunciation` for targeted exercises

## 📊 Performance Metrics

The AI analyzes your conversations across multiple dimensions:

| Metric | Description |
|--------|-------------|
| **Vocabulary** | Range and appropriateness of word choice |
| **Grammar** | Accuracy of sentence structures and conjugations |
| **Fluency** | Natural flow and coherence of speech |
| **Pronunciation** | Clarity and accuracy (inferred from transcription) |
| **Code-Switching** | Frequency of language mixing |

## 🛠️ Technical Stack

**Frontend:**
- React 18 (Hooks)
- TailwindCSS
- Web Audio API
- WebSocket (real-time communication)
- AudioWorklet (low-latency audio processing)

**Backend:**
- FastAPI (async Python web framework)
- WebSocket connections
- Server-sent events (SSE)

**AI/ML Models:**
- OpenAI Whisper (`faster-whisper`)
- Qwen2.5 Claude (API)
- Piper TTS (local inference)
- Argos Translate (local NMT)
- FastText (language detection)

**Infrastructure:**
- Docker & Docker Compose
- Volume persistence for models
- Tailscale (HTTPS/networking)

## 📁 Project Structure
```
.
├── backend/
│   ├── main.py                 # FastAPI app & WebSocket handlers
│   ├── ai_conversation.py      # Claude integration
│   ├── audio_processing.py     # Whisper STT, Piper TTS
│   ├── translation.py          # Argos translate
│   ├── code_switch_detector.py # Language detection
│   └── session_manager.py      # Conversation state
├── frontend/
│   └── index.html              # React SPA
├── docker-compose.yml
├── Dockerfile
└── README.md
```

## 🔧 Configuration

### Environment Variables
```bash
# Required


# Optional
SESSION_TTL_HOURS=2                    # Session expiry
CLEANUP_INTERVAL_SECONDS=300          # Memory cleanup
OMP_NUM_THREADS=4                     # CPU threading
WHISPER_MODEL=medium                  # base/small/medium/large
TTS_VOICE=en_US-lessac-medium         # Piper voice model
```

### Proficiency Levels

| Level | CEFR | HSK | Description |
|-------|------|-----|-------------|
| A1 | A1 | HSK 1 | Absolute beginner |
| A2 | A2 | HSK 2 | Elementary |
| B1 | B1 | HSK 3 | Intermediate |
| B2 | B2 | HSK 4-5 | Upper intermediate |
| C1 | C1 | HSK 5-6 | Advanced |
| C2 | C2 | HSK 6 | Near-native mastery |

## 🐛 Troubleshooting

### "crypto.randomUUID is not a function"
**Cause**: Accessing via HTTP (not localhost or HTTPS)  
**Solution**: Use Tailscale HTTPS or access via `http://localhost:8001`

### Microphone not working
**Cause**: Browser requires HTTPS for microphone access  
**Solution**: Deploy with HTTPS using Tailscale or SSL certificates

### Audio is choppy/distorted
**Cause**: Network latency or CPU constraints  
**Solution**: 
- Reduce Whisper model size (`WHISPER_MODEL=small`)
- Increase `OMP_NUM_THREADS` for better CPU utilization
- Use faster TTS voice model

### No translation appearing
**Cause**: Argos models not downloaded  
**Solution**: Wait for initial model download on first run (check logs)

## 🤝 Contributing

Contributions welcome! Please open issues for bugs or feature requests.

### Development Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Run locally (without Docker)
uvicorn main:app --reload --port 8001

# Run tests
pytest tests/
```

## 📜 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- **OpenAI** - Whisper speech recognition
- **Qwen2.5** - Qwen language model
- **Piper TTS** - Fast, offline text-to-speech
- **Argos Translate** - Privacy-friendly translation
- **Tailscale** - Zero-config HTTPS networking

## 📮 Contact

Questions? Open an issue or reach out at [viibe@tutamail.com]

---

**Built with ❤️ for language learners everywhere**

---

## 🗺️ Roadmap

- [ ] Multi-speaker support (group conversations)
- [ ] Spaced repetition vocabulary trainer
- [ ] Progress tracking over time
- [ ] More language pairs (German, Italian, Japanese, Korean)
- [ ] Mobile app (React Native)
- [ ] Offline mode (local Claude alternative)
- [ ] Grammar exercises generator
- [ ] Voice cloning for consistent AI persona
EOF
```

### Option 2: Using a Text Editor

1. Open your favorite text editor
2. Create a new file named `README.md` in your project root
3. Copy and paste all the content I provided earlier
4. Save the file

### Option 3: Download Directly

I can't create a file for you directly, but you can:

1. Copy the entire markdown content I provided
2. Go to your project directory
3. Create `README.md`
4. Paste the content
5. Save

### Your Project Structure Should Look Like:
```
your-project/
├── README.md          ← Create this file
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .env
├── backend/
│   └── ...
└── frontend/
    └── ...
