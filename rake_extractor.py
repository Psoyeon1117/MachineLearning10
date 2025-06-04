from rake_nltk import Rake
from typing import List, Tuple
import re

class RAKEKeywordExtractor:
    def __init__(self):
        """RAKE 키워드 추출기 초기화"""
        # 커스텀 불용어 리스트 생성
        custom_stopwords = [
            # 기본 영어 불용어
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
            'my', 'your', 'his', 'her', 'its', 'our', 'their',
            'this', 'that', 'these', 'those', 'here', 'there', 'where', 'when', 'why', 'how',
            'not', 'no', 'yes', 'all', 'any', 'some', 'each', 'every', 'other', 'another',
            'more', 'most', 'less', 'much', 'many', 'few', 'little', 'big', 'small',
            'first', 'last', 'next', 'new', 'old', 'good', 'bad', 'right', 'wrong',
            'up', 'down', 'out', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
            
            # 기사/뉴스 관련 흔한 단어
            'said', 'told', 'says', 'tell', 'tells', 'telling',
            'according', 'report', 'reports', 'reported', 'reporting',
            'news', 'article', 'story', 'stories',
            'huffpost', 'facebook', 'instagram', 'twitter', 'social', 'media',
            'post', 'posts', 'posted', 'comment', 'comments',
            'interview', 'interviews', 'speak', 'spoke', 'speaking',
            'former', 'current', 'recent', 'recently', 'now', 'today', 'yesterday',
            'year', 'years', 'month', 'months', 'day', 'days', 'time', 'times',
            'also', 'still', 'just', 'only', 'even', 'though', 'however', 'whether',
            'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
            'want', 'wanted', 'wants', 'need', 'needed', 'needs',
            'like', 'likes', 'look', 'looks', 'looked', 'looking',
            'think', 'thinks', 'thought', 'know', 'knows', 'knew', 'known',
            'feel', 'feels', 'felt', 'feeling', 'make', 'makes', 'made', 'making',
            'go', 'goes', 'went', 'going', 'come', 'comes', 'came', 'coming',
            'get', 'gets', 'got', 'getting', 'take', 'takes', 'took', 'taking',
            'give', 'gives', 'gave', 'given', 'giving',
            'work', 'works', 'worked', 'working', 'call', 'calls', 'called', 'calling',
            'put', 'puts', 'ask', 'asks', 'asked', 'asking',
            'see', 'sees', 'saw', 'seen', 'seeing', 'find', 'finds', 'found', 'finding',
            
            # 의미없는 동사/형용사/부사 추가
            'happen', 'support', 'love', 'express', 'remind', 'draw', 'say',
            'show', 'help', 'try', 'keep', 'let', 'turn', 'become', 'seem',
            'bring', 'follow', 'begin', 'start', 'stop', 'end', 'finish',
            'continue', 'change', 'move', 'run', 'walk', 'talk', 'write', 'read',
            'listen', 'watch', 'play', 'kind', 'sure', 'clear', 'nice', 'great',
            'ever', 'never', 'always', 'often', 'really', 'very', 'quite',
            'soon', 'early', 'quickly', 'slowly', 'properly', 'completely'
        ]
        
        # 더 엄격한 설정으로 RAKE 초기화
        self.rake = Rake(
            stopwords=custom_stopwords,
            min_length=2,  # 최소 2단어
            max_length=3,  # 최대 3단어
            include_repeated_phrases=False
        )
    
    def _preprocess_text(self, text: str) -> str:
        """텍스트 전처리 - 구두점과 문장 구분 개선"""
        if not isinstance(text, str):
            return ""
        
        # 1. 기본 정리
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text.strip())
        
        # 2. 숫자와 연도 처리 (연도는 보존하되 긴 숫자는 제거)
        text = re.sub(r'\b\d{5,}\b', ' ', text)  # 5자리 이상 숫자 제거
        
        # 3. 문장 구분을 위한 구두점 추가
        # 단어 경계에서 대문자가 시작되는 곳에 마침표 추가
        text = re.sub(r'([a-z])([A-Z])', r'\1. \2', text)
        
        # 4. 일반적인 문장 끝 패턴 감지하여 구두점 추가
        # said, told 등 뒤에 오는 내용 구분
        text = re.sub(r'\b(said|told|explained|noted|added|claimed|stated)\s+([a-z])', r'\1. \2', text)
        
        # 5. 년도 뒤에 구두점 추가
        text = re.sub(r'(\d{4})\s+([a-z])', r'\1. \2', text)
        
        # 6. 명사 + 동사 패턴 구분
        text = re.sub(r'([a-z]+)\s+(said|told|work|found|feel|think|know|want|need|make|take|go|come|see|get|give|call|put|ask|tell)', r'\1. \2', text)
        
        return text
    
    def _is_meaningful_keyword(self, phrase: str) -> bool:
        """의미있는 키워드인지 판단"""
        words = phrase.lower().split()
        
        # 너무 일반적인 동사들
        common_verbs = {
            'happen', 'support', 'love', 'express', 'remind', 'draw', 'say', 'tell',
            'show', 'give', 'take', 'make', 'get', 'go', 'come', 'see', 'know',
            'think', 'feel', 'want', 'need', 'help', 'try', 'keep', 'let', 'put',
            'turn', 'become', 'seem', 'look', 'find', 'bring', 'follow', 'begin',
            'start', 'stop', 'end', 'finish', 'continue', 'change', 'move', 'run',
            'walk', 'talk', 'speak', 'write', 'read', 'listen', 'watch', 'play'
        }
        
        # 너무 일반적인 형용사들
        common_adjectives = {
            'good', 'bad', 'big', 'small', 'new', 'old', 'long', 'short', 'high',
            'low', 'right', 'wrong', 'important', 'different', 'same', 'sure',
            'possible', 'available', 'free', 'open', 'close', 'full', 'empty',
            'easy', 'hard', 'simple', 'complex', 'clear', 'kind', 'nice', 'great'
        }
        
        # 너무 일반적인 부사들
        common_adverbs = {
            'ever', 'never', 'always', 'often', 'sometimes', 'usually', 'really',
            'very', 'quite', 'rather', 'pretty', 'soon', 'late', 'early',
            'quickly', 'slowly', 'carefully', 'clearly', 'properly', 'completely'
        }
        
        # 전체 구문이 일반적인 단어들로만 이루어져 있는지 확인
        all_common = common_verbs | common_adjectives | common_adverbs
        
        # 모든 단어가 일반적이면 거부
        if all(word in all_common for word in words):
            return False
        
        # 첫 번째 단어가 동사나 부사면 거부
        if words[0] in (common_verbs | common_adverbs):
            return False
            
        # 마지막 단어가 동사면 거부
        if words[-1] in common_verbs:
            return False
        
        # 적어도 하나의 명사성 단어가 있어야 함
        has_noun = any(word not in all_common for word in words)
        return has_noun
    
    def _filter_keywords(self, keywords: List[Tuple[str, float]], n_keywords: int) -> List[Tuple[str, float]]:
        """키워드 필터링 및 정리"""
        # 추가 불용어 (후처리용)
        additional_stopwords = {
            'told', 'said', 'huffpost', 'facebook', 'instagram', 'according',
            'former', 'model', 'year', 'time', 'also', 'still', 'even', 'one',
            'two', 'new', 'former model', 'model agent', 'last year', 'say brinkman',
            'happen support', 'ever properli', 'draw sure', 'kind', 'support love',
            'express inspir', 'draw kind'
        }
        
        filtered = []
        seen = set()
        
        for phrase, score in keywords:
            # 정리
            clean_phrase = phrase.strip().lower()
            
            # 기본 필터링
            if (len(clean_phrase) >= 6 and  # 최소 6글자
                len(clean_phrase) <= 50 and  # 최대 50글자
                len(clean_phrase.split()) >= 2 and  # 최소 2단어
                len(clean_phrase.split()) <= 3 and  # 최대 3단어
                clean_phrase not in seen and
                clean_phrase not in additional_stopwords and
                not clean_phrase.isdigit() and
                not re.match(r'^[^a-zA-Z]*$', clean_phrase) and  # 알파벳 포함 확인
                not any(stop in clean_phrase for stop in additional_stopwords) and
                self._is_meaningful_keyword(clean_phrase)):  # 의미있는 키워드 체크
                
                filtered.append((phrase, score))
                seen.add(clean_phrase)
                
                if len(filtered) >= n_keywords:
                    break
        
        return filtered
    
    def extract_keywords(self, document: str, n_keywords: int = 5) -> List[Tuple[str, float]]:
        """RAKE를 사용한 키워드 추출"""
        # 전처리
        processed_doc = self._preprocess_text(document)
        
        if not processed_doc.strip():
            return []
        
        try:
            # RAKE 키워드 추출
            self.rake.extract_keywords_from_text(processed_doc)
            
            # 점수와 함께 키워드 가져오기
            keyword_scores = self.rake.get_ranked_phrases_with_scores()
            
            if not keyword_scores:
                return []
            
            # (키워드, 점수) 형태로 변환
            result = [(phrase, score) for score, phrase in keyword_scores]
            
            # 필터링
            filtered_result = self._filter_keywords(result, n_keywords)
            
            return filtered_result
            
        except Exception as e:
            print(f"RAKE 추출 오류: {e}")
            return []
    
    def get_keywords_only(self, document: str, n_keywords: int = 5) -> List[str]:
        """키워드만 반환 (점수 제외)"""
        keyword_scores = self.extract_keywords(document, n_keywords)
        return [keyword for keyword, score in keyword_scores]

if __name__ == "__main__":
    # 테스트 코드
    extractor = RAKEKeywordExtractor()
    test_doc = """
    Machine learning is a subset of artificial intelligence that focuses on algorithms 
    that can learn from data. Deep learning uses neural networks with multiple layers 
    to model complex patterns in large datasets.
    """
    
    keywords = extractor.extract_keywords(test_doc, 5)
    print("RAKE Keywords:")
    for keyword, score in keywords:
        print(f"  {keyword}: {score:.4f}")
