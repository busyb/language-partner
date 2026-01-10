import re
from typing import List, Dict

from utility.UniversalTranslator import UniversalTranslator
from utility.dict_lookup import DictionaryLookup


class CodeSwitchDetector:
    """
    Binary code-switch detector using dictionary lookup.

    Invariant:
    - Exactly TWO languages involved
    - One language is the practicing (main) language
    - The other is the code-switch language
    - English MAY be either, but logic is symmetric
    """

    # Unicode-aware tokenizer
    _TOKEN_RE = re.compile(
        r"[\u4e00-\u9fff]+|[a-zA-Záéíóúñàèìòùâêîôûäëïöü]+",
        re.UNICODE,
    )

    lookup = DictionaryLookup()
    translator = UniversalTranslator()

    # --------------------------------------------------
    # Tokenization
    # --------------------------------------------------
    @classmethod
    def _tokenize(cls, text: str) -> List[str]:
        return cls._TOKEN_RE.findall(text.lower())

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------
    def detect_foreign_words(
        self,
        text: str,
        practicing_lang: str,   # main language
        code_switch_lang: str,  # secondary language
    ) -> List[Dict[str, str]]:
        """
        Detect tokens belonging to code_switch_lang
        when practicing_lang is the expected language.
        """

        text = text.strip()
        if not text:
            return []

        tokens = self._tokenize(text)
        detected: List[Dict[str, str]] = []

        seen = set()
        for token in tokens:
            token_langs = []
            if self.lookup.is_word(token, practicing_lang):
                token_langs.append(practicing_lang)
            if self.lookup.is_word(token, code_switch_lang):
                token_langs.append(code_switch_lang)
            if code_switch_lang in token_langs and practicing_lang not in token_langs:
                if token not in seen:
                    detected.append({"token": token, "lang": code_switch_lang})
                    seen.add(token)

        detected_with_translations = self.translator.translate_token_list(detected, practicing_lang)

        return detected_with_translations



# ======================= TESTS ======================= #

def test_code_switch_detector_binary():
    detector = CodeSwitchDetector()

    tests = [
        # --------------------------------------------------
        # Practicing Spanish, English code-switch
        # --------------------------------------------------
        (
            "Hola, hola, hola, ¿cómo estás?",
            "es",
            "en",
            [],
        ),
        (
            "Estoy learning mucho programming",
            "es",
            "en",
            ["learning", "programming"],
        ),
        (
            "Me gusta el chocolate",
            "es",
            "en",
            [],
        ),
        (
            "Gracias I like this",
            "es",
            "en",
            ["i", "like", "this"],
        ),

        # --------------------------------------------------
        # Practicing English, Spanish code-switch
        # --------------------------------------------------
        (
            "I like pizza",
            "en",
            "es",
            [],
        ),
        (
            "I like chocolate mucho",
            "en",
            "es",
            [],
        ),
        (
            "Hello gracias amigo",
            "en",
            "es",
            ["gracias"], #  "amigo" valid english word
        ),

        # --------------------------------------------------
        # Edge cases
        # --------------------------------------------------
        (
            "",
            "es",
            "en",
            [],
        ),
        (
            "123 !!!",
            "en",
            "es",
            [],
        ),
        (
            "programming programming programming",
            "es",
            "en",
            ["programming"],
        ),
    ]

    print("\n🧪 BINARY CODE-SWITCH DETECTOR TESTS")
    all_passed = True

    for text, practicing_lang, code_switch_lang, expected in tests:
        result = detector.detect_foreign_words(
            text=text,
            practicing_lang=practicing_lang,
            code_switch_lang=code_switch_lang,
        )

        detected = [item["token"] for item in result]

        passed = detected == expected
        status = "✅ PASS" if passed else "❌ FAIL"

        print(
            f"{status} | '{text}' "
            f"(practice={practicing_lang}, switch={code_switch_lang}) "
            f"→ {detected} (expected {expected})"
        )

        if not passed:
            all_passed = False

    print(f"\n🎯 {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")


# -------------------- Run Tests -------------------- #
if __name__ == "__main__":
    test_code_switch_detector_binary()
