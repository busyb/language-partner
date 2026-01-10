# dictionary_lookup.py
import json
from pathlib import Path
from PyMultiDictionary import MultiDictionary


# ---------------- DictionaryLookup ---------------- #
class DictionaryLookup:
    SPANISH_WORDS_TABLE = set()
    FRENCH_WORDS_TABLE = set()
    ENGLISH_WORDS_TABLE = set()

    SPANISH_CITIES = {"madrid", "barcelona", "valencia", "sevilla"}
    FRENCH_CITIES = {"paris", "lyon", "marseille", "toulouse"}
    ENGLISH_CITIES = {"london", "manchester", "birmingham", "edinburgh"}

    # Initialize PyMultiDictionary
    multi_dict = MultiDictionary()

    @classmethod
    def setup_words(cls, spanish_path: str = None, french_path: str = None, english_path: str = None):
        """
        Load word lists for Spanish, French, and English into sets for fast lookup.
        """

        def load_words(path):
            if path is None:
                return set()
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")
            with open(path, encoding="utf-8") as f:
                return {word.lower() for word in json.load(f)}

        BASE_DIR = Path(__file__).parent / "language dictionary"
        cls.SPANISH_WORDS_TABLE = load_words(BASE_DIR / spanish_path)
        cls.FRENCH_WORDS_TABLE = load_words(BASE_DIR / french_path)
        cls.ENGLISH_WORDS_TABLE = load_words(BASE_DIR / english_path)

    def __init__(self,
                 spanish_path="spanish_words.json",
                 french_path="french_words.json",
                 english_path="english_words.json"):
        self.setup_words(spanish_path, french_path, english_path)

    def is_word(self, word: str, lang: str) -> bool:
        """
        Check if a word exists in the dictionary (or as city/proper noun).
        Uses PyMultiDictionary as a backup if not found in tables.
        """
        w = word.lower()
        lang = lang.lower()

        if lang == "es":
            if w in self.SPANISH_WORDS_TABLE or w in self.SPANISH_CITIES:
                return True
        elif lang == "fr":
            if w in self.FRENCH_WORDS_TABLE or w in self.FRENCH_CITIES:
                return True
        elif lang == "en":
            if w in self.ENGLISH_WORDS_TABLE or w in self.ENGLISH_CITIES:
                return True
        elif lang == "zh":
            # Basic Chinese character check
            if any('\u4e00' <= char <= '\u9fff' for char in word):
                return True
            return False
        else:
            raise ValueError(f"Unsupported language: {lang}")

        # Backup lookup with PyMultiDictionary
        try:
            result = self.multi_dict.meaning(word, lang)
            if result:
                return True
        except Exception:
            pass

        return False


# ---------------------- Tests ---------------------- #
def run_tests():
    lookup = DictionaryLookup()

    tests = {
        "en": {
            "positive": ["apple", "programming", "love", "pizza", "chat"],
            "negative": ["我", "hola", "123", "!!"]
        },
        "fr": {
            "positive": ["bonjour", "fromage", "merci", "heureux", "chat"],
            "negative": ["apple", "123", "!!"]
        },
        "es": {
            "positive": ["hola", "gracias", "amigo", "perro", "gato"],
            "negative": ["apple", "hello", "chicken", "123", "!!"]
        },
        "zh": {
            "positive": ["我", "你", "他", "爱", "学习"],
            "negative": ["apple", "skate", "chicken", "123", "!!"]
        }
    }

    for lang, data in tests.items():
        print(f"\nTesting {lang.upper()}")
        for w in data["positive"]:
            assert lookup.is_word(w, lang), f"{w} should be {lang}"
        for w in data["negative"]:
            assert not lookup.is_word(w, lang), f"{w} should NOT be {lang}"
        print(f"✅ {lang.upper()} tests passed")

    print("\n🎯 All dictionary tests passed!")


if __name__ == "__main__":
    run_tests()
