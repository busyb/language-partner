from typing import Dict, List


class SpeakingReportPromptBuilder:
    """
    Builds final LLM prompts for speaking report generation.
    The returned string is sent as-is to the grading LLM.
    """

    SUPPORTED_LANGUAGES = {"english", "french", "spanish", "chinese"}
    SUPPORTED_LEVELS = {"A1", "A2", "B1", "B2", "C1", "C2"}
    LANG_MAP = {
        "en": "English",
        "es": "Spanish",
        "fr": "French",
        "zh": "Chinese",
    }

    HSK_MAPPING = {
        "A1": "HSK 1",
        "A2": "HSK 2",
        "B1": "HSK 3",
        "B2": "HSK 4-5",
        "C1": "HSK 5-6",
        "C2": "HSK 6"
    }

    # -------------------------
    # Feature 1: Conversation Report
    # -------------------------
    def build_conversation_report_prompt(
            self,
            language: str,
            level: str,
            conversation: str,
    ) -> list[dict]:
        """
        Builds a prompt for a conversation-level CEFR assessment.
        Returns a messages array with strict system/user separation.
        """

        if language == "zh":
            level_string = self.HSK_MAPPING.get(level)

        else:
            level_string = f"CEFR {level}"


        language_name = self.LANG_MAP.get(language, language)
        self._validate(language_name, level)

        system_prompt = f"""
    You are a professional language assessment assistant.

    Your task is to analyze a learner's SPOKEN CONVERSATION
    and produce a FINAL REPORT based on CEFR level expectations.

    LANGUAGE: {language_name}
    language LEVEL: {level_string}

    STRICT RULES:
    - Output JSON ONLY
    - Do NOT explain your reasoning
    - Do NOT greet the user
    - Do NOT ask questions
    - Do NOT continue the conversation
    - Do NOT roleplay or act as a tutor
    - Output JSON as a single line without indentation.
    - If you cannot comply, output an empty JSON object {{ }}

    Focus on communication effectiveness, not perfection.
    Adapt expectations to level {level_string}.
    """.strip()

        user_prompt = f"""
    Conversation transcript (chronological):

    {conversation}

    --------------------
    WHAT TO EVALUATE
    --------------------

    Evaluate ONLY at a conversation level:

    1. Comprehensibility (primary)
    2. Fluency & flow
    3. Repeated grammar patterns that affect meaning
    4. Vocabulary range & naturalness
    5. High-level issues (no word scoring)

    --------------------
    OUTPUT SCHEMA (STRICT)
    --------------------

    {{
      "meta": {{
        "language": "{language_name}",
        "level": "{level}",
        "context": "conversation"
      }},
      "overall_scores": {{
        "comprehensibility": 0-100,
        "fluency": 0-100,
        "grammar_effectiveness": 0-100,
        "vocabulary_naturalness": 0-100
      }},
      "strengths": [
        "string",
        "string"
      ],
      "key_issues": [
        {{
          "issue": "string",
          "impact": "low | medium | high",
          "example": "string or null"
        }}
      ],
      "level_alignment": {{
        "meets_expectations": true | false,
        "explanation": "string"
      }},
      "action_plan": {{
        "focus_areas": [
          "string",
          "string"
        ],
        "next_session_goal": "string"
      }}
    }}

    Remember:
    - This is NOT a pronunciation drill
    - Do NOT include per-word feedback
    - Do NOT rewrite the user's sentences
    """.strip()

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    # -------------------------
    # Feature 2: Listen & Repeat Report
    # -------------------------

    def build_pronunciation_drill_report_prompt(
            self,
            language: str,
            drill_results: str,
            learner_level: str = None  # Optional CEFR level for reference
    ) -> list[dict]:
        """
        Builds a prompt for sentence-level listen-and-repeat pronunciation reports.
        """

        if language == "zh":
            level_string = self.HSK_MAPPING.get(learner_level)

        else:
            level_string = f"CEFR {learner_level}"


        language_name = self.LANG_MAP.get(language, language)

        system_prompt = f"""
    You are a professional pronunciation assessment assistant.

    Your task is to analyze LISTEN-AND-REPEAT speaking drills
    and produce a FINAL PRONUNCIATION REPORT based on CEFR-level expectations.

    LANGUAGE: {language_name}
    language LEVEL: {level_string}
    
    STRICT RULES:
    - Output JSON ONLY
    - Do NOT explain your reasoning
    - Do NOT greet the user
    - Do NOT ask questions
    - Do NOT continue the conversation
    - Do NOT roleplay or act as a tutor
    - If you cannot comply, output an empty JSON object {{ }}
    - Be precise and technical
    - Focus on pronunciation, timing, word accuracy, rhythm, and stress
    - Ignore grammar, vocabulary choice, or sentence meaning
    
    """.strip()

        user_prompt = f"""

    Sentence drill results (ordered):
    f{drill_results}



    Each item includes:
    - overall_score
    - per_word_scores
    - word_feedback
    - alignment (insertion, deletion, substitution)
    - heard_text

    --------------------
    WHAT TO EVALUATE
    --------------------

    1. Word accuracy consistency
    2. Common error types (deletions, substitutions, insertions)
    3. Missed or weak words
    4. Rhythm and stress stability
    5. Improvement across attempts

    --------------------
    OUTPUT SCHEMA (STRICT)
    --------------------

    {{
      "meta": {{
        "language": "{language_name}",
        "context": "pronunciation_drill",
        "level": "{learner_level}"
      }},
      "overall_pronunciation_score": 0-100,
      "word_accuracy": {{
        "average": 0-100,
        "most_missed_words": [
          {{
            "word": "string",
            "issue": "missing | unclear | mispronounced"
          }}
        ]
      }},
      "error_patterns": {{
        "deletions": ["string"],
        "substitutions": ["string"],
        "insertions": ["string"]
      }},
      "prosody_feedback": {{
        "rhythm": "good | unstable | poor",
        "stress": "good | unstable | poor",
        "timing_notes": "string"
      }},
      "improvement_trend": {{
        "direction": "improving | stable | declining",
        "evidence": "string"
      }},
      "targeted_drills": [
        {{
          "focus": "string",
          "example": "string"
        }}
      ],
      "next_attempt_focus": "string"
    }}

    Remember:
    - This is NOT a conversation evaluation.
    - Do NOT give grammar or vocabulary advice.
    - Focus strictly on pronunciation, word-level accuracy, prosody, and improvement trends.
    """.strip()

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]



    # -------------------------
    # Validation
    # -------------------------
    def _validate(self, language: str, level: str):
        if language.lower() not in self.SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language: {language}")
        if level not in self.SUPPORTED_LEVELS:
            raise ValueError(f"Unsupported CEFR level: {level}")
