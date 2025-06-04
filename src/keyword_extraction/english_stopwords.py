"""
영어 불용어 및 공통 단어 리스트 (더 포괄적)
"""
import re # Ensure 're' is imported

ENGLISH_STOPWORDS = {
    # 기본 불용어
    'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 
    'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 
    'after', 'above', 'below', 'between', 'among', 'throughout', 'despite',
    'towards', 'upon', 'concerning', 'this', 'that', 'these', 'those',
    
    # 대명사
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 
    'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 
    'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
    'they', 'them', 'their', 'theirs', 'themselves', 
    
    # 의문사
    'what', 'which', 'who', 'whom', 'whose', 'when', 'where', 'why', 'how',
    
    # be 동사
    'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
    
    # 조동사
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 
    'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'shall',
    
    # 일반 동사
    'said', 'say', 'says', 'saying', 'get', 'got', 'getting', 'go', 
    'going', 'goes', 'went', 'come', 'came', 'coming', 'comes', 'take',
    'taken', 'took', 'taking', 'make', 'made', 'making', 'see', 'saw', 'seen',
    'look', 'looked', 'looking', 'use', 'used', 'using', 'find', 'found',
    'give', 'gave', 'given', 'tell', 'told', 'work', 'worked', 'call', 'called',
    
    # 숫자 및 순서
    'one', 'two', 'three', 'first', 'second', 'last', 'next', 'previous',
    
    # 형용사
    'new', 'old', 'good', 'bad', 'big', 'small', 'long', 'short', 'high', 
    'low', 'right', 'left', 'same', 'different', 'each', 'every', 'all',
    'some', 'any', 'many', 'much', 'more', 'most', 'other', 'such', 'only',
    
    # 부사
    'very', 'really', 'just', 'now', 'then', 'here', 'there', 'where',
    'well', 'also', 'too', 'even', 'still', 'again', 'back', 'down',
    'out', 'off', 'over', 'under', 'around', 'away', 'up', 'so', 'no',
    'not', 'yes', 'maybe', 'perhaps', 'quite', 'rather', 'pretty',
    
    # 접속사
    'because', 'since', 'although', 'though', 'however', 'therefore',
    'thus', 'hence', 'while', 'whereas', 'unless', 'until', 'if',
    
    # 기타 일반적인 단어
    'time', 'way', 'day', 'man', 'thing', 'place', 'part', 'kind', 'hand',
    'eye', 'life', 'world', 'case', 'point', 'group', 'company', 'number',
    'fact', 'year', 'week', 'month', 'state', 'question', 'problem', 'area',
    
    # 축약형 및 일반적인 형태
    'don', 'won', 'can', 'isn', 'aren', 'wasn', 'weren', 'haven', 'hasn',
    'hadn', 'doesn', 'didn', 'wouldn', 'couldn', 'shouldn', 'mustn',
}

def is_english_stopword(word: str) -> bool:
    """영어 불용어 체크 (개선된 버전)"""
    return word.lower().strip() in ENGLISH_STOPWORDS

def is_valid_english_keyword(word: str, min_length: int = 3) -> bool:
    """유효한 영어 키워드인지 종합 체크 (완화된 버전)"""
    if not word:
        return False
    
    clean_word = word.strip().lower()
    
    # 1. Length check (uses min_length passed to function, e.g. 2 from midlevel_model)
    if len(clean_word) < min_length:
        return False
    
    # 2. Content validation:
    #    - Must contain at least one letter.
    #    - All characters must be alphanumeric or hyphen.
    #    - Cannot be only hyphens (e.g. "--" is not a keyword).
    #    - Cannot be empty after strip.
    if not clean_word: # Already checked by `if not word` and `len(clean_word) < min_length` but good for safety.
        return False

    has_alpha = False
    valid_chars = True
    all_hyphens = True # Assume all are hyphens until proven otherwise

    for char_val in clean_word:
        if char_val.isalpha():
            has_alpha = True
            all_hyphens = False
        elif char_val.isdigit():
            all_hyphens = False
        elif char_val == '-':
            pass # Hyphens are allowed
        else: # Invalid character encountered
            valid_chars = False
            break 
    
    if not has_alpha or not valid_chars or (all_hyphens and clean_word): # if all_hyphens is still true and clean_word is not empty, it means it was all hyphens.
        return False
        
    # 3. 핵심 불용어만 체크 (완화된 리스트)
    core_stopwords = {
        'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 
        'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 
        'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 
        'did', 'each', 'she', 'use', 'very', 'well', 'were', 'what', 'with'
    }
    
    if clean_word in core_stopwords:
        return False
    
    # 4. 반복 문자 체크 (aaa, bbb 등 제외)
    # This means if all characters in the word are identical (e.g., "aaa", "++")
    # And the word has more than one character (e.g. "a" is not "aa")
    if len(set(clean_word)) == 1 and len(clean_word) > 1:
        return False # Reject words like "aaa" or "---" if they passed content validation somehow (e.g. min_length=1 allowed 'a', which is fine)
                      # The original check was `if len(set(clean_word)) == 1: return False;`.
                      # This rejects single character words if min_length allows them. e.g. 'a'.
                      # However, single char words are usually stopwords or caught by min_length.
                      # The condition `len(clean_word) > 1` ensures "aa", "bb", etc. are caught, but not "a".


    # 5. 너무 긴 단어 제외 (오타 가능성)
    if len(clean_word) > 20:
        return False
    
    return True