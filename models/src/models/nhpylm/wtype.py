from src.models.nhpylm.ctype import *

WORDTYPE_NUM_TYPES: int =  9

WORDTYPE_ALPHABET = 1
WORDTYPE_NUMBER = 2
WORDTYPE_SYMBOL = 3
WORDTYPE_HIRAGANA = 4
WORDTYPE_KATAKANA = 5
WORDTYPE_KANJI = 6
WORDTYPE_KANJI_HIRAGANA = 7
WORDTYPE_KANJI_KATAKANA = 8
WORDTYPE_OTHER = 9

def is_dash(c) -> bool:
  if hex(ord(c)) == 0x30FC:
    return True
  return False

def is_hiragana(c) -> bool:
  t = detect_ctype(c)
  if t == CTYPE_HIRAGANA:
    return True
  # Could also be a dash to indicate a long vowel.
  return is_dash(c)

def is_katakana(c) -> bool:
    t = detect_ctype(c)
    if t == CTYPE_KATAKANA:
        return True
    if t == CTYPE_KATAKANA_PHONETIC_EXTENSIONS:
        return True
    # Could also be a dash to indicate a long vowel.
    return is_dash(c)


def is_kanji(c) -> bool:
    t = detect_ctype(c)
    if t == CTYPE_CJK_UNIFIED_IDEOGRAPHS:
        return True
    if t == CTYPE_CJK_UNIFIED_IDEOGRAPHS_EXTENSION_A:
        return True
    if t == CTYPE_CJK_UNIFIED_IDEOGRAPHS_EXTENSION_B: 
        return True
    if t == CTYPE_CJK_UNIFIED_IDEOGRAPHS_EXTENSION_C:
        return True
    if t == CTYPE_CJK_UNIFIED_IDEOGRAPHS_EXTENSION_D:
        return True
    if t == CTYPE_CJK_UNIFIED_IDEOGRAPHS_EXTENSION_E:
        return True
    if t == CTYPE_CJK_UNIFIED_IDEOGRAPHS_EXTENSION_F:
        return True
    if t == CTYPE_CJK_RADICALS_SUPPLEMENT:
        return True
    return False


def is_number(c) -> bool:
    t = detect_ctype(c)
    character = hex(ord(c))
    if t == CTYPE_BASIC_LATIN:
        if 0x30 <= int(character, 0) and int(character, 0) <= 0x39:
            return True
        return False
    if t == CTYPE_NUMBER_FORMS:
        return True
    if t == CTYPE_COMMON_INDIC_NUMBER_FORMS:
        return True
    if t == CTYPE_AEGEAN_NUMBERS:
        return True
    if t == CTYPE_ANCIENT_GREEK_NUMBERS:
        return True
    if t == CTYPE_COPTIC_EPACT_NUMBERS:
        return True
    if t == CTYPE_SINHALA_ARCHAIC_NUMBERS:
        return True
    if t == CTYPE_CUNEIFORM_NUMBERS_AND_PUNCTUATION:
        return True
    return False


def is_alphabet(c) -> bool:
    character = hex(ord(c))
    if 0x41 <= int(character, 0) and int(character, 0) <= 0x5a:
        return True
    if 0x61 <= int(character, 0) and int(character, 0) <= 0x7a:
        return True
    return False


def is_symbol(c) -> bool:
    if is_alphabet(c): 
        return False
    if is_number(c):
        return False
    if is_kanji(c):
        return False
    if is_hiragana(c): 
        return False 
    return True


def detect_word_type(word):
    return detect_word_type_substr(word, 0, len(word) - 1)


def detect_word_type_substr(chars: list, start: int, end: int):
    num_alphabet = 0
    num_number = 0
    num_symbol = 0
    num_hiragana = 0
    num_katakana = 0
    num_kanji = 0
    num_dash = 0
    size = end - start + 1

    # println!("start is , end is , chars is :?", start, end, chars)
    for i in range(start, end + 1): 
        target = chars[i]
        if is_alphabet(target):
            num_alphabet += 1
            continue
        if is_number(target):
            num_number += 1
            continue
        if is_dash(target): 
            num_dash += 1
            continue
        if is_hiragana(target):
            num_hiragana += 1
            continue
        if is_katakana(target):
            num_katakana += 1
            continue
        if is_kanji(target):
            num_kanji += 1
            continue
        num_symbol += 1
    if num_alphabet == size:
        return WORDTYPE_ALPHABET
    if num_number == size:
        return WORDTYPE_NUMBER
    if num_hiragana + num_dash == size:
        return WORDTYPE_HIRAGANA
    if num_katakana + num_dash == size:
        return WORDTYPE_KATAKANA
    if num_kanji == size:
        return WORDTYPE_KANJI
    if num_symbol == size:
        return WORDTYPE_SYMBOL
    if num_kanji > 0: 
        if num_hiragana + num_kanji == size:
            return WORDTYPE_KANJI_HIRAGANA
        if num_hiragana > 0:
            if num_hiragana + num_kanji + num_dash == size:
                return WORDTYPE_KANJI_HIRAGANA
        if num_katakana + num_kanji == size:
            return WORDTYPE_KANJI_KATAKANA
        if num_katakana > 0:
            if num_katakana + num_kanji + num_dash == size:
                return WORDTYPE_KANJI_KATAKANA
    return WORDTYPE_OTHER