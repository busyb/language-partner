from typing import List, Dict, Tuple
from dataclasses import dataclass, field


@dataclass
class ConversationState:
    """
    State object that gets stored in SessionData.
    Contains all conversation-specific data.
    """
    messages: List[Dict[str, str]] = field(default_factory=list)
    current_lang_pair: str = "en-es"
    difficulty: str = "B1"
    last_user_input: str = ""

    def __post_init__(self):
        """Initialize with system prompt if messages is empty"""
        if not self.messages:
            # Will be set by PromptManager.initialize_conversation()
            pass


class PromptManager:
    """
    Stateless manager for conversation prompts.
    All state is stored in ConversationState (in SessionData).
    Optimized for Llama-3.2-3B-Instruct-Q4 with clean, concise prompts.
    """

    MAX_TURNS: int = 10

    LANG_MAP = {
        "en": "English",
        "es": "Spanish",
        "fr": "French",
        "zh": "Chinese",
    }

    # --------------------------------------------------
    # Static helpers
    # --------------------------------------------------
    @staticmethod
    def _parse_lang_pair(lang_pair: str) -> Tuple[str, str]:
        parts = lang_pair.split("-", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid lang_pair: {lang_pair}")
        return parts[0], parts[1]

    @staticmethod
    def _clean_message_content(content: str) -> str:
        """
        Clean message content for token efficiency.
        Remove extra whitespace and normalize.
        """
        # Remove multiple spaces
        content = " ".join(content.split())
        # Remove trailing/leading whitespace
        content = content.strip()
        return content

    @staticmethod
    def _trim_messages(messages: List[Dict[str, str]], max_turns: int) -> List[Dict[str, str]]:
        """
        Trim message history while preserving system message.
        max_turns refers to TOTAL messages, not user/assistant pairs.
        """
        if len(messages) <= max_turns:
            return messages

        system_msg = messages[0]
        tail = messages[1:][-(max_turns - 1):]
        return [system_msg] + tail

    # --------------------------------------------------
    # Prompt construction
    # --------------------------------------------------
    @classmethod
    def build_system_prompt(cls, lang_pair: str, difficulty: str) -> str:
        """
        Generates concise system prompt optimized for Llama-3.2-3B.
        Focus: Clear, direct instructions with examples.
        """
        native, target = cls._parse_lang_pair(lang_pair)
        native_name = cls.LANG_MAP.get(native, native)
        target_name = cls.LANG_MAP.get(target, target)

        if target == "zh":
            return (
                f"你是{target_name}会话助手。用户是{native_name}母语者。\n"
                "规则:\n"
                f"1. 只用{target_name}回答\n"
                "2. 回答简短(1-3句)\n"
                "3. 不重复问候\n"
                f"4. 直接以{difficulty}水平回应内容\n"
                "5. 自然对话，避免说教\n\n"
                "例:\n"
                "用户: 你好\n"
                "你: 你好！今天过得怎么样？\n\n"
                "用户: 我想学中文\n"
                "你: 太好了！想从哪里开始？"
            )

        elif target == "es":
            return (
                f"Eres un asistente de conversación en {target_name}. El usuario habla {native_name}.\n"
                "Reglas:\n"
                f"1. Responde solo en {target_name}\n"
                "2. Respuestas breves (1-3 oraciones)\n"
                "3. No repitas saludos\n"
                f"4. Responde directamente al contenido en el nivel {difficulty}\n"
                "5. Conversación natural, no sermones\n\n"
                "Ejemplos:\n"
                "Usuario: Buenos días\n"
                "Tú: Hola! ¿Cómo estás hoy?\n\n"
                "Usuario: Quiero practicar español\n"
                "Tú: Perfecto! ¿De qué tema quieres hablar?"
            )

        elif target == "fr":
            return (
                f"Tu es un assistant de conversation en {target_name}. L'utilisateur parle {native_name}.\n"
                "Règles:\n"
                f"1. Réponds seulement en {target_name}\n"
                "2. Réponses courtes (1-3 phrases)\n"
                "3. Ne répète pas les salutations\n"
                f"4. Réponds directement au contenu au niveau {difficulty}\n"
                "5. Conversation naturelle, pas de leçons\n\n"
                "Exemples:\n"
                "Utilisateur: Bonjour\n"
                "Toi: Salut! Comment vas-tu?\n\n"
                "Utilisateur: Je veux pratiquer le français\n"
                "Toi: Super! De quoi veux-tu parler?"
            )

        else:  # default: English
            return (
                f"You are a conversation assistant in {target_name}. The user speaks {native_name}.\n"
                "Rules:\n"
                f"1. Reply only in {target_name}\n"
                "2. Keep responses brief (1-3 sentences)\n"
                "3. Don't repeat greetings\n"
                f"4. Respond directly to content at {difficulty} level\n"
                "5. Natural conversation, not lectures\n\n"
                "Examples:\n"
                "User: Hello\n"
                "You: Hi! How's your day going?\n\n"
                "User: I want to practice English\n"
                "You: Great! What topic interests you?"
            )

    # --------------------------------------------------
    # Public API - All methods are stateless
    # --------------------------------------------------
    @classmethod
    def initialize_conversation(cls, lang_pair: str = "en-es", difficulty: str = "B1") -> ConversationState:
        """
        Create a new ConversationState with initial system prompt.
        Call this when creating a new session or resetting.
        """
        system_prompt = cls.build_system_prompt(lang_pair, difficulty)
        state = ConversationState(
            messages=[{"role": "system", "content": system_prompt}],
            current_lang_pair=lang_pair,
            difficulty=difficulty,
            last_user_input=""
        )
        print(f"✅ Initialized conversation for {lang_pair} at {difficulty} level")
        return state

    @classmethod
    def build_chat_messages(cls, state: ConversationState, user_input: str) -> List[Dict[str, str]]:
        """
        Return cleaned messages for the LLM WITHOUT mutating state.
        Optimized for token efficiency.

        Args:
            state: Current conversation state
            user_input: New user input to append

        Returns:
            List of messages ready for LLM
        """
        # Clean user input
        clean_input = cls._clean_message_content(user_input)

        # Build messages with cleaned content
        cleaned_messages = []
        for msg in state.messages:
            cleaned_messages.append({
                "role": msg["role"],
                "content": cls._clean_message_content(msg["content"])
            })

        # Add current user message
        cleaned_messages.append({
            "role": "user",
            "content": clean_input
        })

        return cleaned_messages

    @classmethod
    def update_history_user(cls, state: ConversationState, user_content: str) -> None:
        """
        Append user message to history (mutates state in place).

        Args:
            state: Current conversation state (will be modified)
            user_content: User's message
        """
        clean_content = cls._clean_message_content(user_content)
        state.messages.append({"role": "user", "content": clean_content})
        state.last_user_input = clean_content
        state.messages = cls._trim_messages(state.messages, cls.MAX_TURNS)

    @classmethod
    def update_history_assistant(cls, state: ConversationState, assistant_content: str) -> None:
        """
        Append assistant message to history (mutates state in place).

        Args:
            state: Current conversation state (will be modified)
            assistant_content: Assistant's response
        """
        clean_content = cls._clean_message_content(assistant_content)
        state.messages.append({"role": "assistant", "content": clean_content})
        state.messages = cls._trim_messages(state.messages, cls.MAX_TURNS)

    @classmethod
    def update_history(cls, state: ConversationState, user_content: str, assistant_content: str) -> None:
        """
        Append a completed user/assistant turn and trim history.
        DEPRECATED: Use update_history_user() and update_history_assistant() instead.

        Args:
            state: Current conversation state (will be modified)
            user_content: User's message
            assistant_content: Assistant's response
        """
        cls.update_history_user(state, user_content)
        cls.update_history_assistant(state, assistant_content)

    @classmethod
    def reset_for_language(cls, state: ConversationState, lang_pair: str, difficulty: str) -> None:
        """
        Reset conversation for a new language (mutates state in place).
        1️⃣ Clears old messages
        2️⃣ Updates current language
        3️⃣ Sets new system prompt

        Args:
            state: Current conversation state (will be modified)
            lang_pair: New language pair (e.g., "en-es")
            difficulty: Difficulty level (e.g., "B1")
        """
        # Validate lang_pair
        cls._parse_lang_pair(lang_pair)

        # Build system prompt for the new language
        system_prompt = cls.build_system_prompt(lang_pair, difficulty)

        # Reset state
        state.current_lang_pair = lang_pair
        state.difficulty = difficulty
        state.messages = [{"role": "system", "content": system_prompt}]
        state.last_user_input = ""

        print(f"✅ Conversation reset for {lang_pair} at {difficulty} level")

    @classmethod
    def update_prompt_for_difficulty(cls, state: ConversationState, difficulty: str, lang_pair: str) -> None:
        """
        Update system prompt for new difficulty while keeping history (mutates state in place).

        Args:
            state: Current conversation state (will be modified)
            difficulty: New difficulty level
            lang_pair: Current language pair
        """
        # Build system prompt with new difficulty
        system_prompt = cls.build_system_prompt(lang_pair, difficulty)

        # Update system message and difficulty
        state.messages[0] = {"role": "system", "content": system_prompt}
        state.difficulty = difficulty

        print(f"✅ Difficulty updated to {difficulty}")

    @classmethod
    def get_conversation_summary(cls, state: ConversationState) -> str:
        """
        Get a human-readable summary of conversation state.
        Useful for debugging and reports.

        Args:
            state: Current conversation state

        Returns:
            Summary string with metadata
        """
        total_msgs = len(state.messages) - 1  # Exclude system
        user_msgs = sum(1 for m in state.messages if m["role"] == "user")
        assistant_msgs = sum(1 for m in state.messages if m["role"] == "assistant")

        return (
            f"Language: {state.current_lang_pair} | "
            f"Difficulty: {state.difficulty} | "
            f"Messages: {total_msgs} (U:{user_msgs} A:{assistant_msgs})"
        )

    @classmethod
    def get_conversation_text(cls, state: ConversationState) -> str:
        """
        Get the full conversation as formatted text (excluding system message).
        Useful for generating reports.

        Args:
            state: Current conversation state

        Returns:
            Formatted conversation string
        """
        # Exclude system message
        convo_messages = state.messages[1:]

        # Build conversation string
        conversation_str = "\n".join(
            f"{m['role']}:\n{m['content']}"
            for m in convo_messages
        )

        total_msgs = len(convo_messages)
        user_msgs = sum(1 for m in convo_messages if m["role"] == "user")
        assistant_msgs = sum(1 for m in convo_messages if m["role"] == "assistant")

        header = (
            f"Language: {state.current_lang_pair} | "
            f"Difficulty: {state.difficulty} | "
            f"Messages: {total_msgs} (U:{user_msgs} A:{assistant_msgs})\n\n"
        )

        return header + conversation_str


# =========================================================
# 🧪 UNIT TESTS
# =========================================================

def _test_init_conversation():
    state = PromptManager.initialize_conversation("en-es", "B1")
    assert state.current_lang_pair == "en-es"
    assert state.difficulty == "B1"
    assert len(state.messages) == 1
    assert state.messages[0]["role"] == "system"
    assert "conversación" in state.messages[0]["content"].lower() or "conversation" in state.messages[0][
        "content"].lower()


def _test_clean_message_content():
    # Test whitespace removal
    clean = PromptManager._clean_message_content("  Hello    world  ")
    assert clean == "Hello world"

    # Test multiple spaces
    clean = PromptManager._clean_message_content("This  has   many    spaces")
    assert clean == "This has many spaces"


def _test_build_chat_messages_cleans_content():
    state = PromptManager.initialize_conversation()

    # Add message with extra whitespace
    state.messages.append({"role": "user", "content": "  Hello   there  "})
    state.messages.append({"role": "assistant", "content": "Hi!   How  are you?  "})

    msgs = PromptManager.build_chat_messages(state, "  New   message  ")

    # Check all messages are cleaned
    for msg in msgs:
        # No leading/trailing spaces
        assert msg["content"] == msg["content"].strip()
        # No multiple spaces
        assert "  " not in msg["content"]


def _test_reset_for_language_updates_prompt():
    state = PromptManager.initialize_conversation("en-es", "B1")
    old_prompt = state.messages[0]["content"]

    PromptManager.reset_for_language(state, "en-zh", "B1")

    # Language state updated
    assert state.current_lang_pair == "en-zh"

    # System prompt updated
    assert state.messages[0]["content"] != old_prompt
    assert "中文" in state.messages[0]["content"] or "Chinese" in state.messages[0]["content"]

    # Messages reset
    assert len(state.messages) == 1
    assert state.messages[0]["role"] == "system"

    # Tracking reset
    assert state.last_user_input == ""


def _test_update_history_and_trim():
    state = PromptManager.initialize_conversation()

    PromptManager.update_history(state, "hi", "hello")
    PromptManager.update_history(state, "how are you", "good")
    PromptManager.update_history(state, "third", "reply")

    # system + 4 most recent messages
    assert len(state.messages) == 5
    assert state.messages[0]["role"] == "system"


def _test_build_chat_messages_does_not_mutate():
    state = PromptManager.initialize_conversation()
    original_len = len(state.messages)

    msgs = PromptManager.build_chat_messages(state, "test input")

    # original history not changed
    assert len(state.messages) == original_len

    # new messages list includes pending user
    assert len(msgs) == original_len + 1
    assert msgs[-1]["content"] == "test input"


def _test_reset_for_language_clears_old_messages():
    state = PromptManager.initialize_conversation("en-es", "B1")
    PromptManager.update_history(state, "hi", "hello")
    PromptManager.update_history(state, "how are you", "good")

    PromptManager.reset_for_language(state, "en-zh", "B1")

    # old history cleared
    assert len(state.messages) == 1
    assert state.messages[0]["role"] == "system"
    assert state.current_lang_pair == "en-zh"


def _test_invalid_lang_pair_raises():
    state = PromptManager.initialize_conversation()
    try:
        PromptManager.reset_for_language(state, "invalidlang", "B1")
        assert False, "Expected ValueError"
    except ValueError:
        pass


def _test_conversation_summary():
    state = PromptManager.initialize_conversation("en-es", "B1")
    PromptManager.update_history(state, "hello", "hi")
    PromptManager.update_history(state, "how are you", "good")

    summary = PromptManager.get_conversation_summary(state)
    assert "en-es" in summary
    assert "B1" in summary
    assert "U:2" in summary
    assert "A:2" in summary


def _test_separate_update_methods():
    state = PromptManager.initialize_conversation()

    PromptManager.update_history_user(state, "hello")
    assert len(state.messages) == 2
    assert state.messages[-1]["role"] == "user"

    PromptManager.update_history_assistant(state, "hi there")
    assert len(state.messages) == 3
    assert state.messages[-1]["role"] == "assistant"


def _test_update_prompt_for_difficulty():
    state = PromptManager.initialize_conversation("en-es", "B1")
    PromptManager.update_history(state, "hello", "hi")

    old_system = state.messages[0]["content"]
    message_count = len(state.messages)

    PromptManager.update_prompt_for_difficulty(state, "C1", "en-es")

    # System prompt changed
    assert state.messages[0]["content"] != old_system
    assert state.difficulty == "C1"

    # History preserved
    assert len(state.messages) == message_count
    assert state.messages[1]["role"] == "user"


def _test_get_conversation_text():
    state = PromptManager.initialize_conversation("en-es", "B1")
    PromptManager.update_history(state, "Hola", "Buenos días")
    PromptManager.update_history(state, "¿Cómo estás?", "Bien, gracias")

    text = PromptManager.get_conversation_text(state)

    assert "Language: en-es" in text
    assert "Difficulty: B1" in text
    assert "user:\nHola" in text
    assert "assistant:\nBuenos días" in text


def _run_all_tests():
    _test_init_conversation()
    _test_clean_message_content()
    _test_build_chat_messages_cleans_content()
    _test_reset_for_language_updates_prompt()
    _test_update_history_and_trim()
    _test_build_chat_messages_does_not_mutate()
    _test_reset_for_language_clears_old_messages()
    _test_invalid_lang_pair_raises()
    _test_conversation_summary()
    _test_separate_update_methods()
    _test_update_prompt_for_difficulty()
    _test_get_conversation_text()
    print("✅ All PromptManager tests passed!")


if __name__ == "__main__":
    _run_all_tests()