
WORDTYPE_NUM_TYPES: int = 1

WORDTYPE_MELODY = 1

def detect_word_type(word):
    return detect_word_type_substr(word, 0, len(word) - 1)


def detect_word_type_substr(chars: list, start: int, end: int):
    return WORDTYPE_MELODY