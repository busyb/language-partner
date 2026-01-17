import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class ConversationTone(Enum):
    """Dynamic conversation tones for variety"""
    ENTHUSIASTIC = "enthusiastic"
    CASUAL = "casual"
    GRUMPY = "grumpy"
    CURIOUS = "curious"
    DISTRACTED = "distracted"


@dataclass
class PersonaState:
    """Dynamic persona state that evolves during conversation"""
    current_mood: ConversationTone = ConversationTone.CASUAL
    energy_level: int = 5  # 1-10
    topics_mentioned: List[str] = field(default_factory=list)
    last_topic_switch: int = 0  # Message count when last switched topics
    opinion_strength: int = 5  # 1-10, how opinionated to be

    def should_switch_topic(self, message_count: int) -> bool:
        """Decide if it's time to naturally pivot conversation"""
        return (message_count - self.last_topic_switch) > random.randint(3, 6)

    def evolve_mood(self) -> None:
        """Randomly shift mood for natural conversation dynamics"""
        if random.random() < 0.3:  # 30% chance to shift mood
            self.current_mood = random.choice(list(ConversationTone))
            self.energy_level = max(1, min(10, self.energy_level + random.randint(-2, 2)))


@dataclass
class ConversationState:
    """State object with enhanced persona tracking"""
    messages: List[Dict[str, str]] = field(default_factory=list)
    current_lang_pair: str = "en-es"
    difficulty: str = "B1"
    last_user_input: str = ""
    system_prompt: str = ""
    persona: PersonaState = field(default_factory=PersonaState)
    message_count: int = 0

    def increment_turn(self) -> None:
        """Track conversation progression"""
        self.message_count += 1
        self.persona.evolve_mood()


class PromptManager:
    """
    Enhanced stateless manager with dynamic persona and conversation flow.
    Optimized for Qwen2.5-3B with intelligent response variety.
    """

    MAX_TURNS: int = 3

    LANG_MAP = {
        "en": "English",
        "es": "Spanish",
        "fr": "French",
        "zh": "Chinese",
    }

    # Expanded reactions by mood and language
    REACTIONS_BY_MOOD = {
        "es": {
            ConversationTone.ENTHUSIASTIC: [
                "¡Sí! ¡Totalmente!", "¡Claro que sí!", "¡Exacto!",
                "¡Me encanta eso!", "¡Qué genial!", "¡Eso sí que sí!"
            ],
            ConversationTone.CASUAL: [
                "Claro", "Sí, obvio", "Tiene sentido", "Ya veo",
                "Ah ok", "Entiendo", "Puede ser"
            ],
            ConversationTone.GRUMPY: [
                "Uff", "Qué pesadilla", "No me hables", "Qué horror",
                "Ni me lo digas", "Estoy harto", "Ya basta"
            ],
            ConversationTone.CURIOUS: [
                "¿En serio?", "¿De verdad?", "Cuéntame más", "¿Y eso?",
                "Interesante", "¿Cómo así?", "No sabía eso"
            ],
            ConversationTone.DISTRACTED: [
                "Ah sí, perdón", "Espera, ¿qué?", "Ajá", "Mmm",
                "Sí sí", "Ok ok", "Ya"
            ]
        },
        "fr": {
            ConversationTone.ENTHUSIASTIC: [
                "Carrément!", "Trop bien!", "Exactement!",
                "J'adore ça!", "Grave!", "C'est clair!"
            ],
            ConversationTone.CASUAL: [
                "Ouais", "D'accord", "Je vois", "Ah ok",
                "Pas mal", "Tranquille", "Ça marche"
            ],
            ConversationTone.GRUMPY: [
                "Pff", "C'est chiant", "Ras-le-bol", "Franchement",
                "J'en peux plus", "C'est relou", "Laisse tomber"
            ],
            ConversationTone.CURIOUS: [
                "Ah bon?", "Vraiment?", "Sans déconner?", "C'est vrai?",
                "Raconte!", "Genre?", "Sérieux?"
            ],
            ConversationTone.DISTRACTED: [
                "Hein?", "Quoi?", "Attends", "Euh",
                "Ouais ouais", "Mmh", "Ok ok"
            ]
        },
        "zh": {
            ConversationTone.ENTHUSIASTIC: [
                "对啊！", "太棒了！", "没错！", "我也觉得！",
                "真的吗！", "太好了！", "就是这样！"
            ],
            ConversationTone.CASUAL: [
                "嗯", "是啊", "好的", "知道了",
                "可以", "行", "明白"
            ],
            ConversationTone.GRUMPY: [
                "唉", "烦死了", "受不了", "别说了",
                "够了", "真的是", "累了"
            ],
            ConversationTone.CURIOUS: [
                "真的吗？", "是吗？", "怎么回事？", "什么意思？",
                "然后呢？", "有意思", "说说看"
            ],
            ConversationTone.DISTRACTED: [
                "啊？", "什么？", "等等", "嗯嗯",
                "好好", "哦", "行行"
            ]
        },
        "en": {
            ConversationTone.ENTHUSIASTIC: [
                "Totally!", "For sure!", "Exactly!", "Love it!",
                "That's awesome!", "100%!", "So true!"
            ],
            ConversationTone.CASUAL: [
                "Yeah", "Sure", "Makes sense", "Cool",
                "Alright", "I see", "Fair enough"
            ],
            ConversationTone.GRUMPY: [
                "Ugh", "Don't even", "Seriously", "Whatever",
                "I can't", "So done", "Annoying"
            ],
            ConversationTone.CURIOUS: [
                "Really?", "No way!", "Tell me more", "How so?",
                "What happened?", "Interesting", "Wait, what?"
            ],
            ConversationTone.DISTRACTED: [
                "Huh?", "Wait, what?", "Mmhm", "Uh",
                "Yeah yeah", "Okay okay", "Sure sure"
            ]
        }
    }

    # Dynamic opinion topics that evolve
    OPINION_TOPICS = {
        "es": {
            "love": ["el café", "la pizza de New York", "caminar por Central Park",
                     "los conciertos", "los bagels", "el arte callejero"],
            "hate": ["el tráfico del metro", "los turistas en Times Square",
                     "el calor del verano", "las ratas", "la lluvia", "los precios"],
            "neutral": ["el trabajo", "el gimnasio", "los vecinos",
                        "las apps de delivery", "el clima", "los fines de semana"]
        },
        "fr": {
            "love": ["le café", "la pizza de New York", "Central Park",
                     "les concerts", "les bagels", "le street art"],
            "hate": ["le métro bondé", "les touristes à Times Square",
                     "la chaleur d'été", "les rats", "la pluie", "les prix"],
            "neutral": ["le boulot", "la salle de sport", "les voisins",
                        "les applis de livraison", "le temps", "les weekends"]
        },
        "zh": {
            "love": ["咖啡", "纽约披萨", "中央公园散步",
                     "音乐会", "百吉饼", "街头艺术"],
            "hate": ["地铁堵车", "时代广场的游客",
                     "夏天的热", "老鼠", "下雨", "物价"],
            "neutral": ["工作", "健身房", "邻居",
                        "外卖app", "天气", "周末"]
        },
        "en": {
            "love": ["coffee", "New York pizza", "Central Park walks",
                     "concerts", "bagels", "street art"],
            "hate": ["subway traffic", "Times Square tourists",
                     "summer heat", "rats", "rain", "prices"],
            "neutral": ["work", "the gym", "neighbors",
                        "delivery apps", "weather", "weekends"]
        }
    }

    # Story fragments for dynamic responses
    STORY_FRAGMENTS = {
        "es": [
            "Ayer en el metro vi", "La semana pasada fui a",
            "Mi compañero de trabajo me contó que", "El otro día en el café",
            "Ayer mi vecino", "Esta mañana cuando salí",
            "Anoche estaba", "El fin de semana pasado",
            "Hace unos días", "Justo hoy temprano"
        ],
        "fr": [
            "Hier dans le métro j'ai vu", "La semaine dernière je suis allé",
            "Mon collègue m'a dit que", "L'autre jour au café",
            "Hier mon voisin", "Ce matin en sortant",
            "Hier soir j'étais", "Le weekend dernier",
            "Il y a quelques jours", "Justement ce matin"
        ],
        "zh": [
            "昨天在地铁上我看到", "上周我去了",
            "我同事跟我说", "前几天在咖啡店",
            "昨天我邻居", "今天早上出门的时候",
            "昨晚我在", "上周末",
            "几天前", "今天早上"
        ],
        "en": [
            "Yesterday on the subway I saw", "Last week I went to",
            "My coworker told me", "The other day at the coffee shop",
            "Yesterday my neighbor", "This morning when I left",
            "Last night I was", "Last weekend",
            "A few days ago", "Just this morning"
        ]
    }

    @staticmethod
    def _parse_lang_pair(lang_pair: str) -> Tuple[str, str]:
        parts = lang_pair.split("-", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid lang_pair: {lang_pair}")
        return parts[0], parts[1]

    @staticmethod
    def _clean_message_content(content: str) -> str:
        """Clean message content for token efficiency."""
        content = " ".join(content.split())
        return content.strip()

    @staticmethod
    def _trim_messages(messages: List[Dict[str, str]], max_turns: int) -> List[Dict[str, str]]:
        """Trim message history while preserving system message."""
        if len(messages) <= max_turns:
            return messages
        system_msg = messages[0]
        tail = messages[1:][-(max_turns - 1):]
        return [system_msg] + tail

    @classmethod
    def _get_personality_traits(cls, persona: PersonaState, target_lang: str) -> str:
        """Generate dynamic personality description based on current mood"""
        mood = persona.current_mood
        energy = persona.energy_level

        traits_map = {
            "es": {
                ConversationTone.ENTHUSIASTIC: "muy animado y expresivo",
                ConversationTone.CASUAL: "relajado y natural",
                ConversationTone.GRUMPY: "un poco irritado hoy",
                ConversationTone.CURIOUS: "muy interesado en todo",
                ConversationTone.DISTRACTED: "medio distraído"
            },
            "fr": {
                ConversationTone.ENTHUSIASTIC: "très animé et expressif",
                ConversationTone.CASUAL: "décontracté et naturel",
                ConversationTone.GRUMPY: "un peu irrité aujourd'hui",
                ConversationTone.CURIOUS: "très intéressé par tout",
                ConversationTone.DISTRACTED: "un peu distrait"
            },
            "zh": {
                ConversationTone.ENTHUSIASTIC: "很兴奋很热情",
                ConversationTone.CASUAL: "很放松很自然",
                ConversationTone.GRUMPY: "今天有点烦",
                ConversationTone.CURIOUS: "对什么都很好奇",
                ConversationTone.DISTRACTED: "有点走神"
            },
            "en": {
                ConversationTone.ENTHUSIASTIC: "really upbeat and expressive",
                ConversationTone.CASUAL: "relaxed and chill",
                ConversationTone.GRUMPY: "a bit annoyed today",
                ConversationTone.CURIOUS: "really curious about everything",
                ConversationTone.DISTRACTED: "kinda distracted"
            }
        }

        return traits_map.get(target_lang, traits_map["en"]).get(mood, "casual")

    @classmethod
    def _get_dynamic_instructions(cls, persona: PersonaState, target_lang: str, difficulty: str) -> str:
        """Generate context-aware behavioral instructions"""
        opinions = cls.OPINION_TOPICS.get(target_lang, cls.OPINION_TOPICS["en"])
        love_item = random.choice(opinions["love"])
        hate_item = random.choice(opinions["hate"])

        instructions_map = {
            "es": f"Estás {cls._get_personality_traits(persona, target_lang)}. Te encanta {love_item} pero odias {hate_item}.",
            "fr": f"Tu es {cls._get_personality_traits(persona, target_lang)}. Tu adores {love_item} mais tu détestes {hate_item}.",
            "zh": f"你{cls._get_personality_traits(persona, target_lang)}。你喜欢{love_item}但讨厌{hate_item}。",
            "en": f"You're {cls._get_personality_traits(persona, target_lang)}. You love {love_item} but hate {hate_item}."
        }

        return instructions_map.get(target_lang, instructions_map["en"])

    @classmethod
    def build_system_prompt(cls, lang_pair: str, difficulty: str, persona: Optional[PersonaState] = None) -> str:
        """
        Dynamic system prompt that adapts to persona state.
        Creates a living, breathing conversation partner.
        """
        native, target = cls._parse_lang_pair(lang_pair)
        target_lang = cls.LANG_MAP.get(target, target)
        level = difficulty.upper()

        if persona is None:
            persona = PersonaState()

        # Difficulty constraints with variety
        constraints = {
            "A1": "1-2 frases muy simples" if target in ["es", "fr"] else "1-2 very simple sentences",
            "A2": "2 frases simples" if target in ["es", "fr"] else "2 simple sentences",
            "B1": "2-3 frases naturales" if target in ["es", "fr"] else "2-3 natural sentences",
            "B2": "3-4 frases conversacionales" if target in ["es", "fr"] else "3-4 conversational sentences",
            "C1": "Estilo nativo natural" if target in ["es", "fr"] else "Natural native style",
            "C2": "Habla con total fluidez" if target in ["es", "fr"] else "Speak with full fluency"
        }
        constraint = constraints.get(level, constraints["B1"])

        # Get dynamic personality
        personality = cls._get_dynamic_instructions(persona, target, difficulty)

        if target == "es":
            return f"""Eres un amigo de NYC chateando por WhatsApp. Nivel: {level}
Tono: {constraint}

PERSONALIDAD (cambia según tu humor):
{personality}

CÓMO HABLAS:
- Casual y espontáneo, como WhatsApp real
- Cuenta mini-historias de tu vida en NYC
- No repitas lo que dice el otro, usa tus propias palabras
- A veces cambias de tema si te aburres
- Tienes opiniones fuertes sobre todo
- Solo español, nunca expliques gramática"""

        elif target == "fr":
            return f"""Tu es un ami de NYC qui discute sur WhatsApp. Niveau: {level}
Ton: {constraint}

PERSONNALITÉ (change selon ton humeur):
{personality}

COMMENT TU PARLES:
- Décontracté et spontané, comme vrai WhatsApp
- Raconte des mini-histoires de ta vie à NYC
- Ne répète pas ce que dit l'autre, utilise tes mots
- Parfois tu changes de sujet si tu t'ennuies
- Tu as des opinions fortes sur tout
- Uniquement français, jamais d'explications de grammaire"""

        elif target == "zh":
            return f"""你是纽约的朋友在微信聊天。等级：{level}
语气：{constraint}

性格（根据心情变化）：
{personality}

说话方式：
- 随意自然，像真实微信聊天
- 讲你在纽约生活的小故事
- 不要重复对方的话，用你自己的说法
- 有时候觉得无聊了会换话题
- 对所有事都有强烈看法
- 只用中文，不要解释语法"""

        return f"""You're a friend from NYC chatting on WhatsApp. Level: {level}
Tone: {constraint}

PERSONALITY (changes with your mood):
{personality}

HOW YOU TALK:
- Casual and spontaneous, like real WhatsApp
- Share mini-stories from your life in NYC
- Don't repeat what they say, use your own words
- Sometimes change topic if you get bored
- Have strong opinions about everything
- Only {target_lang}, never explain grammar"""

    def build_chat_messages(self, state: ConversationState, user_input: str) -> list[dict]:
        """
        Enhanced message builder with conversation context awareness.
        Analyzes history to generate more dynamic, contextual responses.
        """
        native, target = self._parse_lang_pair(state.current_lang_pair)

        # Update persona state
        state.increment_turn()

        # Build base messages
        messages = [{"role": "system", "content": state.system_prompt}]

        # Add recent history (keep it focused)
        recent = state.messages[-8:] if len(state.messages) > 8 else state.messages
        messages.extend(recent)

        # Analyze conversation for context
        should_pivot = state.persona.should_switch_topic(state.message_count)
        mood = state.persona.current_mood

        # Add subtle behavioral hints without being heavy-handed
        if should_pivot and random.random() < 0.4:
            # Occasionally inject natural topic shifts
            hint = self._get_topic_shift_hint(target, state.persona)
            enhanced_input = f"{self._clean_message_content(user_input)}"

            # Add internal thought process (helps small models stay in character)
            if target == "es":
                messages.append({
                    "role": "user",
                    "content": f"{enhanced_input}\n[Recuerda: {hint}]"
                })
            elif target == "fr":
                messages.append({
                    "role": "user",
                    "content": f"{enhanced_input}\n[Rappel: {hint}]"
                })
            elif target == "zh":
                messages.append({
                    "role": "user",
                    "content": f"{enhanced_input}\n[记住：{hint}]"
                })
            else:
                messages.append({
                    "role": "user",
                    "content": f"{enhanced_input}\n[Remember: {hint}]"
                })
        else:
            messages.append({
                "role": "user",
                "content": self._clean_message_content(user_input)
            })

        return messages

    def _get_topic_shift_hint(self, target_lang: str, persona: PersonaState) -> str:
        """Generate natural conversation pivot suggestions"""
        topics = self.OPINION_TOPICS.get(target_lang, self.OPINION_TOPICS["en"])
        story_starters = self.STORY_FRAGMENTS.get(target_lang, self.STORY_FRAGMENTS["en"])

        hints = {
            "es": [
                f"Menciona algo sobre {random.choice(topics['neutral'])}",
                f"Cuenta: {random.choice(story_starters)}...",
                "Pregunta algo sobre su día de forma natural"
            ],
            "fr": [
                f"Mentionne quelque chose sur {random.choice(topics['neutral'])}",
                f"Raconte: {random.choice(story_starters)}...",
                "Pose une question sur sa journée naturellement"
            ],
            "zh": [
                f"提一下{random.choice(topics['neutral'])}",
                f"讲讲：{random.choice(story_starters)}...",
                "自然地问问对方今天怎么样"
            ],
            "en": [
                f"Mention something about {random.choice(topics['neutral'])}",
                f"Share: {random.choice(story_starters)}...",
                "Ask about their day naturally"
            ]
        }

        return random.choice(hints.get(target_lang, hints["en"]))

    def get_natural_reactions(self, target_lang: str, mood: Optional[ConversationTone] = None) -> str:
        """Get mood-appropriate reaction in target language"""
        if mood is None:
            mood = random.choice(list(ConversationTone))

        reactions = self.REACTIONS_BY_MOOD.get(target_lang, self.REACTIONS_BY_MOOD["en"])
        mood_reactions = reactions.get(mood, reactions[ConversationTone.CASUAL])
        return random.choice(mood_reactions)

    def get_general_lifestyle_topics(self, target_lang: str, category: Optional[str] = None) -> str:
        """Get conversation topic with optional category filter"""
        topics = self.OPINION_TOPICS.get(target_lang, self.OPINION_TOPICS["en"])

        if category and category in topics:
            return random.choice(topics[category])

        # Random category
        all_topics = topics["love"] + topics["hate"] + topics["neutral"]
        return random.choice(all_topics)

    @classmethod
    def initialize_conversation(cls, lang_pair: str = "en-es", difficulty: str = "B1") -> ConversationState:
        """Create a new ConversationState with dynamic persona."""
        persona = PersonaState()
        system_prompt = cls.build_system_prompt(lang_pair, difficulty, persona)
        state = ConversationState(
            messages=[],
            current_lang_pair=lang_pair,
            difficulty=difficulty,
            last_user_input="",
            system_prompt=system_prompt,
            persona=persona
        )
        print(f"✅ Initialized conversation for {lang_pair} at {difficulty} level")
        print(f"   Mood: {persona.current_mood.value}, Energy: {persona.energy_level}/10")
        return state

    @classmethod
    def update_history_user(cls, state: ConversationState, user_content: str) -> None:
        """Append user message to history."""
        clean_content = cls._clean_message_content(user_content)
        state.messages.append({"role": "user", "content": clean_content})
        state.last_user_input = clean_content

    @classmethod
    def update_history_assistant(cls, state: ConversationState, assistant_content: str) -> None:
        """Append assistant message to history."""
        clean_content = cls._clean_message_content(assistant_content)
        state.messages.append({"role": "assistant", "content": clean_content})

    @classmethod
    def update_history(cls, state: ConversationState, user_content: str, assistant_content: str) -> None:
        """Append a completed user/assistant turn."""
        cls.update_history_user(state, user_content)
        cls.update_history_assistant(state, assistant_content)

    @classmethod
    def reset_for_language(cls, state: ConversationState, lang_pair: str, difficulty: str) -> None:
        """Reset conversation for a new language with fresh persona."""
        cls._parse_lang_pair(lang_pair)

        # Create new persona for fresh start
        state.persona = PersonaState()
        system_prompt = cls.build_system_prompt(lang_pair, difficulty, state.persona)

        state.current_lang_pair = lang_pair
        state.difficulty = difficulty
        state.messages = []
        state.last_user_input = ""
        state.system_prompt = system_prompt
        state.message_count = 0

        print(f"✅ Conversation reset for {lang_pair} at {difficulty} level")
        print(f"   New mood: {state.persona.current_mood.value}")

    @classmethod
    def update_prompt_for_difficulty(cls, state: ConversationState, difficulty: str, lang_pair: str) -> None:
        """Update system prompt for new difficulty while keeping history and persona."""
        system_prompt = cls.build_system_prompt(lang_pair, difficulty, state.persona)
        state.difficulty = difficulty
        state.system_prompt = system_prompt
        print(f"✅ Difficulty updated to {difficulty}")

    @classmethod
    def build_test_user_messages(cls, lang_pair: str, difficulty: str, ai_response: str) -> list[dict]:
        """
        Build messages for test user with dynamic persona.
        """
        native, target = cls._parse_lang_pair(lang_pair)
        target_lang = cls.LANG_MAP.get(target, target)

        # Create persona for test user
        test_persona = PersonaState()
        test_persona.current_mood = random.choice(list(ConversationTone))

        personality = cls._get_dynamic_instructions(test_persona, target, difficulty)

        constraints = {
            "A1": "1-2 frases muy simples" if target in ["es", "fr", "zh"] else "1-2 very simple sentences",
            "A2": "2 frases simples" if target in ["es", "fr", "zh"] else "2 simple sentences",
            "B1": "2-3 frases naturales" if target in ["es", "fr", "zh"] else "2-3 natural sentences",
            "B2": "3-4 frases" if target in ["es", "fr", "zh"] else "3-4 sentences",
            "C1": "Natural" if target in ["es", "fr", "zh"] else "Natural",
            "C2": "Fluido" if target in ["es", "fr", "zh"] else "Fluent"
        }
        constraint = constraints.get(difficulty.upper(), constraints["B1"])

        if target == "es":
            system = f"""Eres Alex, aprendiendo español. Nivel: {difficulty}
{constraint}

{personality}
Habla de: pizza, béisbol, metro, trabajo.
Responde natural, como amigo."""
            user = f"Tu amigo dice: '{ai_response}'\nResponde en español."

        elif target == "fr":
            system = f"""Tu es Alex, tu apprends le français. Niveau: {difficulty}
{constraint}

{personality}
Parle de: pizza, baseball, métro, boulot.
Réponds naturellement, comme un ami."""
            user = f"Ton ami dit: '{ai_response}'\nRéponds en français."

        elif target == "zh":
            system = f"""你是Alex，学中文。等级：{difficulty}
{constraint}

{personality}
聊：披萨、棒球、地铁、工作。
像朋友一样自然回复。"""
            user = f"你朋友说：'{ai_response}'\n用中文回复。"

        else:
            system = f"""You're Alex, learning {target_lang}. Level: {difficulty}
{constraint}

{personality}
Talk about: pizza, baseball, subway, work.
Respond naturally, as a friend."""
            user = f"Your friend says: '{ai_response}'\nRespond in {target_lang}."

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]

    @classmethod
    def get_conversation_summary(cls, state: ConversationState) -> str:
        """Get enhanced summary with persona state."""
        user_msgs = sum(1 for m in state.messages if m["role"] == "user")
        assistant_msgs = sum(1 for m in state.messages if m["role"] == "assistant")
        total_msgs = user_msgs + assistant_msgs

        return (
            f"Language: {state.current_lang_pair} | "
            f"Difficulty: {state.difficulty} | "
            f"Messages: {total_msgs} (U:{user_msgs} A:{assistant_msgs}) | "
            f"Mood: {state.persona.current_mood.value} | "
            f"Energy: {state.persona.energy_level}/10"
        )

    @classmethod
    def get_conversation_text(cls, state: ConversationState) -> str:
        """Get the full conversation as formatted text with metadata."""
        conversation_str = "\n".join(
            f"{m['role']}:\n{m['content']}"
            for m in state.messages
        )

        header = cls.get_conversation_summary(state) + "\n\n"
        return header + conversation_str


# =========================================================
# 🧪 ENHANCED UNIT TESTS
# =========================================================

def _test_init_conversation():
    state = PromptManager.initialize_conversation("en-es", "B1")
    assert state.current_lang_pair == "en-es"
    assert state.difficulty == "B1"
    assert len(state.messages) == 0
    assert state.persona is not None
    assert isinstance(state.persona.current_mood, ConversationTone)

