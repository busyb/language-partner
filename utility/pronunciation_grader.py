# utils/pronunciation_grader.py
import difflib
import json
import re
from typing import List, Dict, Tuple, Optional
import numpy as np

LANGUAGE_SUBSTITUTIONS = {
    "es": {
        'th': ['t', 'd', 's'],  # "the" → "te", "de", "se"
        'r': ['l', ''],  # "brown" → "blown", dropped r
        'z': ['s'],  # "lazy" → "lassy"
        'j': ['h', 'x'],  # "jumps" → "humps"
        'v': ['b'],  # "over" → "ober"
        'ch': ['sh'],  # "quick" context
        'b': ['v'],  # Spanish b/v confusion
        'll': ['y', 'j'],  # "pollo" variations
    },
    "fr": {
        'h': [''],  # French often drops 'h'
        'th': ['t', 'd', 'z'],  # "th" → "t", "d", or "z"
        'w': ['v', 'ou'],  # English w → French v or ou
        'r': [''],  # French guttural r often dropped
    },
    "zh": {
        'r': ['l'],  # English r → L in Chinese
        'th': ['s', 't', 'z'],  # English th → s/t/z
        'v': ['w', 'f'],  # English v → w or f
        'l': ['r'],  # L/R confusion
    },
    "en": {
        # For non-native English speakers
        'th': ['t', 'd', 's', 'z'],
        'v': ['b', 'w', 'f'],
        'r': ['l'],
    }
}


class PronunciationGrader:
    """
    Grades pronunciation by comparing expected text to Whisper transcription.
    Handles multiple languages and common L1 interference patterns.
    """

    _grade_history: List[Dict] = []

    SUPPORTED_LANGUAGES = {"en", "es", "fr", "zh"}

    @staticmethod
    def normalize_word(word: str) -> str:
        """Remove punctuation and lowercase."""
        # Remove all punctuation including Spanish ¿¡
        cleaned = re.sub(r'[^\w\s]', '', word)
        return cleaned.strip().lower()

    @classmethod
    def is_likely_substitution(cls, exp_word: str, pred_word: str, language: str) -> bool:
        """
        Check if predicted word matches common substitution patterns.
        Uses edit distance + phonetic pattern matching.
        """
        if pred_word is None or language not in LANGUAGE_SUBSTITUTIONS:
            return False

        exp_lower = exp_word.lower()
        pred_lower = pred_word.lower()

        # Quick exact match
        if exp_lower == pred_lower:
            return True

        # Check edit distance (must be close)
        from difflib import SequenceMatcher
        similarity = SequenceMatcher(None, exp_lower, pred_lower).ratio()
        if similarity < 0.5:
            return False  # Too different to be a substitution

        # Check phonetic patterns
        subs = LANGUAGE_SUBSTITUTIONS.get(language, {})
        for pattern, alts in subs.items():
            if pattern in exp_lower:
                for alt in alts:
                    # Check if substitution exists in predicted
                    if alt == '':
                        # Check if pattern was dropped
                        if pattern not in pred_lower and exp_lower.replace(pattern, '') == pred_lower:
                            return True
                    elif alt in pred_lower:
                        return True

        return False

    @staticmethod
    def calculate_phonetic_similarity(word1: str, word2: str) -> float:
        """
        Calculate phonetic similarity between two words.
        Returns 0-1 score based on character overlap and position.
        """
        if not word1 or not word2:
            return 0.0

        matcher = difflib.SequenceMatcher(None, word1.lower(), word2.lower())
        return matcher.ratio()

    @staticmethod
    def word_level_alignment(expected_words: List[str], predicted_words: List[str]) -> Tuple[List[Dict], List[str]]:
        """
        Align expected and predicted word lists using SequenceMatcher.
        Returns: (aligned_words, extra_words)
        """
        matcher = difflib.SequenceMatcher(None, expected_words, predicted_words)
        aligned = []
        extra_words = []

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                for i, j in zip(range(i1, i2), range(j1, j2)):
                    aligned.append({
                        "exp": expected_words[i],
                        "pred": predicted_words[j],
                        "type": "match"
                    })
            elif tag == "replace":
                # Handle unequal replacement lengths
                exp_count = i2 - i1
                pred_count = j2 - j1

                for i in range(i1, i2):
                    j = j1 + min(i - i1, pred_count - 1)
                    pred_word = predicted_words[j] if j < j2 else None
                    aligned.append({
                        "exp": expected_words[i],
                        "pred": pred_word,
                        "type": "substitution"
                    })
            elif tag == "delete":
                for i in range(i1, i2):
                    aligned.append({
                        "exp": expected_words[i],
                        "pred": None,
                        "type": "deletion"
                    })
            elif tag == "insert":
                for j in range(j1, j2):
                    extra_words.append(predicted_words[j])

        return aligned, extra_words

    @classmethod
    def grade_words(
            cls,
            segments: List[Dict],
            expected_text: str,
            language: str,
            heard: str,
            counter: int
    ) -> Dict:
        """
        Main grading function.

        Args:
            segments: Whisper output with word-level timestamps
                     Can be full result dict or list of segments
            expected_text: Target sentence
            language: ISO language code (en, es, fr, zh)
            heard: Full transcribed text (for display)

        Returns:
            Dictionary with scores, feedback, and alignment info
        """
        if language not in cls.SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language: {language}. Supported: {cls.SUPPORTED_LANGUAGES}")

        # Handle both formats: full Whisper result or segments list
        if isinstance(segments, dict) and 'segments' in segments:
            # Full Whisper result format
            segments = segments['segments']
        elif not isinstance(segments, list):
            raise ValueError(f"Invalid segments format: {type(segments)}")

        # Extract and normalize words from segments
        all_words = []
        for segment in segments:
            words_list = segment.get('words', [])

            # Handle case where segment is a word dict itself
            if not words_list and 'word' in segment:
                words_list = [segment]

            for word_info in words_list:
                word = cls.normalize_word(word_info.get('word', ''))
                if word:  # Skip empty words
                    all_words.append({
                        'word': word,
                        'confidence': word_info.get('probability', word_info.get('confidence', 0.9)),
                        'start': word_info.get('start', 0),
                        'end': word_info.get('end', 0)
                    })

        # If no words extracted, try parsing heard_text directly
        if not all_words and heard:
            print(f"⚠️ No words in segments, parsing heard_text: {heard}")
            # Fallback: use heard text with default confidence
            for word in heard.split():
                normalized = cls.normalize_word(word)
                if normalized:
                    all_words.append({
                        'word': normalized,
                        'confidence': 0.9,
                        'start': 0,
                        'end': 0
                    })

        # Aggregate confidence scores for repeated words
        word_confidences = {}
        for w in all_words:
            word_confidences.setdefault(w['word'], []).append(w['confidence'])

        # Normalize expected and predicted word lists
        expected_words = [cls.normalize_word(w) for w in expected_text.split() if cls.normalize_word(w)]
        predicted_words = [w['word'] for w in all_words]

        print(f"🔍 Expected: {expected_words}")
        print(f"🔍 Predicted: {predicted_words}")

        # Perform alignment
        aligned, extra_words = cls.word_level_alignment(expected_words, predicted_words)

        word_scores = {}
        word_feedback = {}

        for item in aligned:
            # Convert substitution with None to deletion
            if item["type"] == "substitution" and item["pred"] is None:
                item["type"] = "deletion"

            exp_word = item["exp"]
            pred_word = item["pred"]
            atype = item["type"]

            # === Base Score Calculation ===
            if atype == "match":
                base_score = 100
            elif atype == "substitution":
                # Check if it's a common L1 interference pattern
                if cls.is_likely_substitution(exp_word, pred_word, language):
                    # Give partial credit based on phonetic similarity
                    similarity = cls.calculate_phonetic_similarity(exp_word, pred_word)
                    base_score = 30 + (similarity * 40)  # 30-70 range for common errors
                else:
                    # Calculate partial credit for phonetic similarity
                    similarity = cls.calculate_phonetic_similarity(exp_word, pred_word)
                    base_score = similarity * 50  # 0-50 range for other substitutions
            elif atype == "deletion":
                base_score = 0
            else:
                base_score = 0

            # === Confidence Score ===
            avg_conf = 0
            if pred_word and pred_word in word_confidences:
                confidences = word_confidences[pred_word]
                avg_conf = (sum(confidences) / len(confidences)) * 100

            # === Final Weighted Score ===
            # 70% base accuracy, 30% ASR confidence
            final_score = 0.7 * base_score + 0.3 * avg_conf
            word_scores[exp_word] = round(final_score, 1)

            # === Generate Feedback ===
            if final_score < 60:  # Threshold for feedback
                if pred_word is None:
                    word_feedback[exp_word] = "missing"
                elif cls.is_likely_substitution(exp_word, pred_word, language):
                    word_feedback[exp_word] = f"accent: {pred_word}"
                else:
                    word_feedback[exp_word] = f"heard: {pred_word}"

        # Calculate overall score
        overall_score = np.mean(list(word_scores.values())) if word_scores else 0

        result = {
            'overall_score': round(overall_score, 1),
            'per_word_scores': word_scores,
            'word_feedback': word_feedback,
            'extra_words': extra_words,
            'alignment': aligned,
            'heard_text': heard,
            'counter': counter
        }

        # 🔹 Record grading result
        cls._grade_history.append(result)

        return result


    def get_grade_history(cls) -> List[Dict]:
        """
        Return all stored grade_words() results without resetting.
        """
        return cls._grade_history.copy()



    def reset_grade_history(cls):
        """
        Return all stored grade_words() results and reset storage.
        """
        cls._grade_history.clear()

    from typing import List, Dict

    def get_test_pronunciation_results(cls) -> str:
        """
        Returns a list of 11 hard-coded pronunciation drill result objects.
        Each object mimics realistic slight errors in the same structure.
        """
        return cls.clean_objects_for_llm([
            {
                'overall_score': 0.4,
                'per_word_scores': {'la': 0.4, 'puerta': 0.4, 'está': 0.4, 'abierta': 0.4},
                'word_feedback': {'la': 'heard: yo', 'puerta': 'heard: yo', 'está': 'heard: yo',
                                  'abierta': 'heard: yo'},
                'extra_words': [],
                'alignment': [
                    {'exp': 'la', 'pred': 'yo', 'type': 'substitution'},
                    {'exp': 'puerta', 'pred': 'yo', 'type': 'substitution'},
                    {'exp': 'está', 'pred': 'yo', 'type': 'substitution'},
                    {'exp': 'abierta', 'pred': 'yo', 'type': 'substitution'}
                ],
                'heard_text': 'Yo.',
                'counter': 1
            },
            {
                'overall_score': 0.5,
                'per_word_scores': {'la': 0.5, 'puerta': 0.4, 'está': 0.5, 'abierta': 0.4},
                'word_feedback': {'la': 'heard: la', 'puerta': 'heard: yo', 'está': 'heard: esta',
                                  'abierta': 'heard: abierta'},
                'extra_words': ['el'],
                'alignment': [
                    {'exp': 'puerta', 'pred': 'yo', 'type': 'substitution'},
                    {'exp': 'está', 'pred': 'esta', 'type': 'substitution'}
                ],
                'heard_text': 'la yo esta abierta.',
                'counter': 2
            },
            {
                'overall_score': 0.45,
                'per_word_scores': {'la': 0.5, 'puerta': 0.45, 'está': 0.4, 'abierta': 0.45},
                'word_feedback': {'la': 'heard: la', 'puerta': 'heard: puerta', 'está': 'heard: esta',
                                  'abierta': 'heard: abieta'},
                'extra_words': [],
                'alignment': [
                    {'exp': 'está', 'pred': 'esta', 'type': 'substitution'},
                    {'exp': 'abierta', 'pred': 'abieta', 'type': 'substitution'}
                ],
                'heard_text': 'la puerta esta abieta.',
                'counter': 3
            },
            {
                'overall_score': 0.6,
                'per_word_scores': {'la': 0.6, 'puerta': 0.55, 'está': 0.6, 'abierta': 0.55},
                'word_feedback': {'la': 'heard: la', 'puerta': 'heard: puera', 'está': 'heard: está',
                                  'abierta': 'heard: abierta'},
                'extra_words': [],
                'alignment': [
                    {'exp': 'puerta', 'pred': 'puera', 'type': 'substitution'}
                ],
                'heard_text': 'la puera está abierta.',
                'counter': 4
            },
            {
                'overall_score': 0.35,
                'per_word_scores': {'la': 0.3, 'puerta': 0.4, 'está': 0.35, 'abierta': 0.4},
                'word_feedback': {'la': 'heard: lo', 'puerta': 'heard: puerta', 'está': 'heard: está',
                                  'abierta': 'heard: abierta'},
                'extra_words': [],
                'alignment': [
                    {'exp': 'la', 'pred': 'lo', 'type': 'substitution'}
                ],
                'heard_text': 'lo puerta está abierta.',
                'counter': 5
            }
        ])


    def clean_objects_for_llm(cls, obj_list: List[Dict]) -> str:
        """
        Converts list of dicts into one LLM-ready string.
        Each object on a new line, no outer brackets.
        """
        return "\n".join(json.dumps(obj, separators=(',', ':'), ensure_ascii=False) for obj in obj_list)

# =========================================================
# 🧪 UNIT TESTS
# =========================================================

def test_normalize_word():
    grader = PronunciationGrader()
    assert grader.normalize_word("¿Tienes") == "tienes"
    assert grader.normalize_word("hambre?") == "hambre"
    assert grader.normalize_word("helado.") == "helado"
    assert grader.normalize_word("  test!  ") == "test"
    print("✅ test_normalize_word passed")


def test_word_level_alignment():
    expected = ["hello", "world"]
    predicted = ["hello", "word"]
    aligned, extra = PronunciationGrader.word_level_alignment(expected, predicted)

    assert len(aligned) == 2
    assert aligned[0]["type"] == "match"
    assert aligned[1]["type"] == "substitution"
    assert extra == []
    print("✅ test_word_level_alignment passed")


def test_is_likely_substitution_spanish():
    grader = PronunciationGrader()

    # Spanish b/v confusion
    assert grader.is_likely_substitution("vida", "bida", "es") == True

    # Not a substitution
    assert grader.is_likely_substitution("hello", "goodbye", "es") == False

    print("✅ test_is_likely_substitution_spanish passed")


def test_grade_words_spanish():
    segments = [{
        'words': [
            {'word': 'tienes', 'probability': 0.95, 'start': 0.0, 'end': 0.5},
            {'word': 'ambre', 'probability': 0.80, 'start': 0.6, 'end': 1.0},  # Missing 'h'
            {'word': 'te', 'probability': 0.98, 'start': 1.1, 'end': 1.3},
            {'word': 'gustaria', 'probability': 0.92, 'start': 1.4, 'end': 1.8},
            {'word': 'un', 'probability': 0.96, 'start': 1.9, 'end': 2.0},
            {'word': 'helado', 'probability': 0.94, 'start': 2.1, 'end': 2.5},
        ]
    }]

    expected = "¿Tienes hambre? ¿Te gustaría un helado?"
    heard = "tienes ambre te gustaria un helado"

    result = PronunciationGrader.grade_words(segments, expected, "es", heard)

    assert result['overall_score'] > 0
    assert 'tienes' in result['per_word_scores']
    assert 'hambre' in result['per_word_scores']
    assert result['per_word_scores']['tienes'] > result['per_word_scores']['hambre']

    print("✅ test_grade_words_spanish passed")
    print(f"   Overall score: {result['overall_score']}")
    print(f"   Word scores: {result['per_word_scores']}")
    print(f"   Result : {result}")


def test_phonetic_similarity():
    grader = PronunciationGrader()

    # Very similar
    assert grader.calculate_phonetic_similarity("hello", "helo") > 0.8

    # Somewhat similar
    similarity = grader.calculate_phonetic_similarity("hambre", "ambre")
    assert 0.6 < similarity < 0.95

    # Not similar
    assert grader.calculate_phonetic_similarity("hello", "goodbye") < 0.4

    print("✅ test_phonetic_similarity passed")


def run_all_tests():
    test_normalize_word()
    test_word_level_alignment()
    test_is_likely_substitution_spanish()
    test_phonetic_similarity()
    test_grade_words_spanish()
    print("\n🎉 All tests passed!")


if __name__ == "__main__":
    run_all_tests()