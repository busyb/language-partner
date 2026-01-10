"""
Multi-Language Translation Utility
Offline-first using Argos Translate
Supports English, Spanish, French, Chinese
Extensible architecture for future providers
"""
from deep_translator import GoogleTranslator
from typing import List, Dict, Optional, Tuple
from argostranslate import translate
import threading


class UniversalTranslator:
    """
    Extensible translation utility for single words/tokens.
    Supports eager or lazy loading of translation models.
    """

    def __init__(
        self,
        supported_pairs: Optional[List[Tuple[str, str]]] = None,
        preload: bool = True,
        strict: bool = False,
    ):
        """
        Args:
            supported_pairs: list of tuples, e.g. [("en", "es"), ("es", "en")]
            preload: Load all models at startup
            strict: Raise errors instead of returning None
        """
        self.supported_pairs = supported_pairs or [("en", "es"), ("es", "en")]
        self.strict = strict

        self._lock = threading.Lock()
        self._installed_langs = {}
        self._translators: Dict[Tuple[str, str], translate.Translation] = {}

        self._refresh_installed_languages()

        if preload:
            for pair in self.supported_pairs:
                self._load_pair(*pair)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _refresh_installed_languages(self):
        """Cache installed Argos languages"""
        self._installed_langs = {
            lang.code: lang for lang in translate.get_installed_languages()
        }

    def _load_pair(self, src: str, tgt: str):
        """Load translator for a specific language pair"""
        key = (src, tgt)
        if key in self._translators:
            return

        with self._lock:
            if key in self._translators:
                return

            src_lang = self._installed_langs.get(src)
            tgt_lang = self._installed_langs.get(tgt)

            if not src_lang or not tgt_lang:
                raise RuntimeError(f"Missing model {src}→{tgt}. Make sure it's installed.")

            self._translators[key] = src_lang.get_translation(tgt_lang)
            print(f"✅ Loaded {src}→{tgt}")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def add_language_pair(self, src: str, tgt: str):
        pair = (src, tgt)
        if pair not in self.supported_pairs:
            self.supported_pairs.append(pair)
            self._load_pair(src, tgt)

    def get_available_pairs(self) -> List[Tuple[str, str]]:
        return list(self._translators.keys())

    def translate_words(
        self,
        words: List[str],
        src_lang: str,
        tgt_lang: str,
    ) -> List[Dict[str, Optional[str]]]:
        key = (src_lang, tgt_lang)

        try:
            self._load_pair(src_lang, tgt_lang)
        except Exception as e:
            if self.strict:
                raise
            return [
                {
                    "original": w,
                    "translation": None,
                    "error": str(e),
                    "src_lang": src_lang,
                    "tgt_lang": tgt_lang,
                }
                for w in words
            ]

        translator = self._translators[key]

        unique_words = set(words)
        cache: Dict[str, Optional[str]] = {}

        for word in unique_words:
            try:
                cache[word] = translator.translate(word)
            except Exception:
                cache[word] = None
                if self.strict:
                    raise

        return [
            {
                "original": w,
                "translation": cache[w],
                "src_lang": src_lang,
                "tgt_lang": tgt_lang,
            }
            for w in words
        ]

    def translate_single(
        self, word: str, src_lang: str, tgt_lang: str
    ) -> Optional[str]:
        result = self.translate_words([word], src_lang, tgt_lang)
        return result[0]["translation"]

    def translate_token_list(
        self, tokens: List[Dict[str, str]], target_lang: str
    ) -> List[Dict[str, str]]:
        """Translate a list of token dicts and add 'translation'"""
        if not tokens:
            return tokens

        src_lang = tokens[0]["lang"]
        words = [t["token"] for t in tokens]
        translations = self.translate_words(words, src_lang, target_lang)

        for t, trans in zip(tokens, translations):
            t["translation"] = trans["translation"]
            t["trans_lang"] = target_lang

        return tokens

    def translate_sentence(self, sentence: str, src_lang: str, tgt_lang: str) -> Dict[str, str]:
        """
        Translate full sentence using deep_translator (Google Translate).
        Perfect for full sentence translation with proper grammar/context.
        """

        # Normalize Chinese language codes
        if src_lang.lower() == "zh":
            src_lang = "zh-CN"
        if tgt_lang.lower() == "zh":
            tgt_lang = "zh-CN"

        try:
            translator = GoogleTranslator(source=src_lang, target=tgt_lang)
            translated = translator.translate(sentence)  # ✅ Handles full sentences!

            return {
                "text": translated,
                "text_language": tgt_lang.upper()  # Frontend expects "EN", "ES", etc.
            }
        except Exception as e:
            return {
                "text": f"Translation unavailable: {sentence}",
                "text_language": tgt_lang.upper()
            }

# ------------------------------------------------------------------ #
# Main script
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    # -----------------------------
    # 1. Initialize translator
    # -----------------------------
    pairs_to_load = [
        ("en", "es"), ("es", "en"),
        ("en", "fr"), ("fr", "en"),
        ("en", "zh"), ("zh", "en")
    ]

    translator = UniversalTranslator(
        supported_pairs=pairs_to_load,
        preload=True
    )

    # -----------------------------
    # 2. Sample tokens
    # -----------------------------
    tokens = [
        {"token": "cooking", "lang": "en"},
        {"token": "hello", "lang": "en"},
        {"token": "world", "lang": "en"},
    ]

    # -----------------------------
    # 3. Translate to multiple languages
    # -----------------------------
    for target in ["es", "fr", "zh"]:
        print(f"\nTranslations to {target}:")
        translated_tokens = translator.translate_token_list(tokens.copy(), target)
        print("tranlation tokens: ", translated_tokens)
        for t in translated_tokens:
            print(f"{t['token']} → {t['translation']}")
