from rake_nltk import Rake
from typing import List, Tuple
import re

class RAKEKeywordExtractor:
    def __init__(self):
        """RAKE 키워드 추출기 초기화"""
        # 더 엄격한 설정으로 RAKE 초기화
        self.rake = Rake(
            stopwords='english',
            min_length=1,
            max_length=3,  # 최대 3단어로 제한
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
    
    def _filter_keywords(self, keywords: List[Tuple[str, float]], n_keywords: int) -> List[Tuple[str, float]]:
        """키워드 필터링 및 정리"""
        filtered = []
        seen = set()
        
        for phrase, score in keywords:
            # 정리
            clean_phrase = phrase.strip().lower()
            
            # 필터링 조건
            if (len(clean_phrase) >= 3 and  # 최소 3글자
                len(clean_phrase) <= 50 and  # 최대 50글자
                len(clean_phrase.split()) <= 3 and  # 최대 3단어
                clean_phrase not in seen and
                not clean_phrase.isdigit() and
                not re.match(r'^[^a-zA-Z]*$', clean_phrase)):  # 알파벳 포함 확인
                
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
