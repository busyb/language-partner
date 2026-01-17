import json
import random
from pathlib import Path


class LLMStreamer:
    """
    Abstracts streaming from different LLM backends (llama_cpp, Ollama, etc.)
    and supports warm-up. Always yields tokens for streaming; caller aggregates.
    Can skip internal '<think>' reasoning if desired.
    """
    def __init__(self, llm=None, llm_type="llama_cpp", stream_model=None, chat_fn=None, skip_think=True):
        """
        llm: model instance for llama_cpp
        llm_type: "llama_cpp" or "ollama"
        stream_model: Ollama model identifier (for chat())
        chat_fn: Ollama chat function if you want to inject it
        skip_think: whether to skip '<think>' reasoning blocks
        """
        self.llm = llm
        self.llm_type = llm_type
        self.stream_model = stream_model
        self.chat_fn = chat_fn
        self.skip_think = skip_think

    def build_warmup_prompt_generic(self) -> str:
        """
        Generates a fully dynamic generic warm-up prompt for the LLM.
        Runs at startup, before the language is selected.
        Uses <think> so the model can prepare reasoning capacity.
        Prepares the model to handle any target language flexibly and adaptively.
        """
        return (
            "<think>"
            "Do not repeat phrases or questions you already asked in previous turns"
            "Hello! Let's get ready to be a playful, curious, and supportive language partner. "
            "Practice responding naturally in any language the user might choose. "
            "Be prepared to adapt to different language structures, styles, and vocabulary. "
            "Focus on creating engaging, short, and interactive conversations that feel dynamic and responsive. "
            "Mix mostly familiar words with a few slightly challenging ones to gently expand the user's vocabulary, "
            "adjusting the difficulty based on the user's responses. "
            "Notice patterns in user input and adapt tone, phrasing, and questions to keep the user motivated, curious, and confident. "
            "Celebrate attempts, encourage exploration, and maintain smooth, enjoyable dialogue in every language."
            "</think>"
        )



    # -----------------------------
    # WARM-UP
    # -----------------------------
    def warm_up(self):
        """
        Sends a request to pre-load kernels, KV cache, and tokenizer.
        Uses the dynamic warm-up prompt to prepare the LLM for multi-language,
        playful, adaptive conversation.
        Should be called at startup.
        """
        # Use the fully dynamic warm-up prompt
        warmup_prompt = self.build_warmup_prompt_generic()

        # Wrap in a test message as user input
        test_message = [{"role": "user", "content": warmup_prompt}]

        try:
            if self.llm_type == "llama_cpp":
                _ = self.llm.create_chat_completion(
                    messages=test_message,
                    stream=False,
                    temperature=0.6
                )
            elif self.llm_type == "ollama":
                if self.chat_fn is None or self.stream_model is None:
                    raise ValueError("Ollama chat function and model must be provided")
                _ = self.chat_fn(
                    model=self.stream_model,
                    messages=test_message,
                    stream=False
                )
            else:
                raise NotImplementedError(f"Warm-up not implemented for {self.llm_type}")

            print("🔥 LLM warm-up successful!")
        except Exception as e:
            print(f"⚠️ LLM warm-up failed: {e}")

    # -----------------------------
    # STREAMING
    # -----------------------------
    def stream(self, messages):
        """
        Yields text tokens from the model, one by one.
        Caller aggregates for full response.
        Skips '<think>' blocks if skip_think=True.
        """
        inside_think = False

        if self.llm_type == "llama_cpp":
            stream = self.llm.create_chat_completion(messages=messages, stream=True, temperature=1, frequency_penalty=0.7)
            for chunk in stream:
                delta = chunk["choices"][0]["delta"].get("content")
                if not delta:
                    continue

                if self.skip_think:
                    # Handle <think> ... </think>
                    if "<think>" in delta:
                        inside_think = True
                        delta = delta.split("<think>", 1)[0]
                    if "</think>" in delta:
                        inside_think = False
                        delta = delta.split("</think>", 1)[1]
                    if inside_think:
                        continue

                yield delta

        elif self.llm_type == "ollama":
            if self.chat_fn is None or self.stream_model is None:
                raise ValueError("Ollama chat function and model must be provided")
            stream = self.chat_fn(model=self.stream_model, messages=messages, stream=True)
            for chunk in stream:
                delta = chunk["message"]["content"]
                if not delta:
                    continue

                if self.skip_think:
                    if "<think>" in delta:
                        inside_think = True
                        delta = delta.split("<think>", 1)[0]
                    if "</think>" in delta:
                        inside_think = False
                        delta = delta.split("</think>", 1)[1]
                    if inside_think:
                        continue

                yield delta

        else:
            raise NotImplementedError(f"Streaming not implemented for {self.llm_type}")

    def generate_sentence_llm(self, difficulty: str, lang: str):
        """
        Generate a sentence for pronunciation practice based on language and difficulty.

        Args:
            difficulty (str): One of "A1", "A2", "B1", "B2", "C1", "C2"
            lang (str): Language code, e.g., "en", "es", "fr", "zh"

        Returns:
            str: Generated sentence
        """

        # Build prompt dynamically
        prompt = (
            f"Please generate one sentence suitable for pronunciation practice "
            f"in {lang} at the {difficulty} level. "
            f"The sentence should be clear, natural, and appropriate for speaking practice."
            f"respond with only the sentence in {lang} language"
        )

        # Wrap prompt in chat message format for Qwen2.5-3B-instruct
        messages = [
            {"role": "user", "content": prompt}
        ]

        # Call the LLM
        response = self.call_llm_not_stream(messages)

        # Extract the text from the LLM response
        if isinstance(response, dict) and "choices" in response:
            return response["choices"][0]["message"]["content"].strip()
        else:
            return str(response).strip()

    def respondToAI(self, messages):
        """
        Generate Alex's response to the AI's message.
        Now uses the optimized PromptManager method.
        """
        # Get the language pair (assuming you have this info)
        # Build messages using PromptManager

        # Call your LLM
        response = self.call_llm_not_stream(messages)

        # Standardizing response extraction
        if isinstance(response, dict) and "choices" in response:
            content = response["choices"][0]["message"]["content"].strip()
        else:
            content = str(response).strip()

        # Cleaning: Sometimes small models wrap responses in quotes
        return content.replace('"', '').replace('Alex:', '').strip()

    def get_general_lifestyle_topics(self) -> str:
        """
        Returns a list of 10 general conversation topics to keep
        the dialogue dynamic and natural.
        """
        topics = [
            "querer ir a un concierto de rock este fin de semana",
            "el estreno de una película de terror en el cine",
            "ganas de ir a un museo de arte moderno",
            "planear una cena en un restaurante tailandés",
            "necesidad de comprar zapatillas nuevas para caminar",
            "el último libro que empezaste a leer anoche",
            "un podcast de crímenes reales que te tiene enganchado",
            "ganas de salir a bailar salsa o merengue",
            "la pereza de tener que ir al gimnasio más tarde",
            "el plan de hacer una excursión o senderismo el domingo"
        ]

        # Returning a random choice is best for the 3B model's focus
        return random.choice(topics)


    def call_llm_not_stream(self, messages, temperature: float = 1):

        return self.llm.create_chat_completion(frequency_penalty=0.6,
            messages=messages,
            stream=False,
            temperature=temperature  # Uses passed value or 0.6
        )["choices"][0]["message"]["content"]

    def generate_sentence_file(self, lang: str) -> str:
        """
        Retrieve a pronunciation practice sentence from a local file.
        """

        # ---- Normalize inputs ----
        difficulty = random.choice(["A1", "A2", "B1", "B2", "C1", "C2"]).upper()
        lang = lang.lower()

        # ---- Initialize cache on self ----
        if not hasattr(self, "_sentence_cache"):
            sentence_file = Path(__file__).parent.parent / "utility" / "sentences.json"

            if not sentence_file.exists():
                raise FileNotFoundError(
                    "sentences.json not found. Expected structure: "
                    "{lang: {level: [sentences...]}}"
                )

            with open(sentence_file, "r", encoding="utf-8") as f:
                self._sentence_cache = json.load(f)

        data = self._sentence_cache

        # ---- Validation ----
        if lang not in data:
            raise ValueError(f"Unsupported language: {lang}")

        if difficulty not in data[lang]:
            raise ValueError(f"Unsupported difficulty: {difficulty}")

        sentences = data[lang][difficulty]

        if not isinstance(sentences, list) or not sentences:
            raise ValueError(f"No sentences available for {lang} {difficulty}")

        # ---- Random selection ----
        return random.choice(sentences)


