import yake
from typing import List, Tuple
import re

class YAKEKeywordExtractor:
    def __init__(self):
        """YAKE 키워드 추출기 초기화"""
        self.extractor = None
    
    def _preprocess_text(self, text: str) -> str:
        """텍스트 전처리"""
        if not isinstance(text, str):
            return ""
        
        # 다중 공백을 단일 공백으로
        text = re.sub(r'\s+', ' ', text.strip())
        
        return text
    
    def extract_keywords(self, document: str, n_keywords: int = 5) -> List[Tuple[str, float]]:
        """
        YAKE를 사용한 키워드 추출
        
        Args:
            document: 입력 문서
            n_keywords: 추출할 키워드 수
            
        Returns:
            (키워드, 점수) 튜플 리스트 (점수가 낮을수록 좋음)
        """
        # 전처리
        processed_doc = self._preprocess_text(document)
        
        if not processed_doc.strip():
            return []
        
        try:
            # YAKE 키워드 추출기 설정
            kw_extractor = yake.KeywordExtractor(
                lan="en",                    # 언어 설정
                n=3,                        # n-gram 최대 크기
                dedupLim=0.7,               # 중복 제거 임계값
                top=n_keywords * 2,         # 여유있게 더 많이 추출 후 필터링
                features=None
            )
            
            # 키워드 추출
            keywords = kw_extractor.extract_keywords(processed_doc)
            
            # 결과 필터링 및 정리
            filtered_keywords = []
            seen_keywords = set()
            
            for keyword, score in keywords:
                # 키워드 정리
                clean_keyword = keyword.strip().lower()
                
                # 중복 체크 및 최소 길이 체크
                if (clean_keyword not in seen_keywords and 
                    len(clean_keyword) > 2 and 
                    not clean_keyword.isdigit()):
                    
                    filtered_keywords.append((keyword, score))
                    seen_keywords.add(clean_keyword)
                    
                    if len(filtered_keywords) >= n_keywords:
                        break
            
            return filtered_keywords
            
        except Exception as e:
            print(f"YAKE 키워드 추출 중 오류: {e}")
            return []
    
    def get_keywords_only(self, document: str, n_keywords: int = 5) -> List[str]:
        """
        키워드만 반환 (점수 제외)
        
        Args:
            document: 입력 문서
            n_keywords: 추출할 키워드 수
            
        Returns:
            키워드 리스트
        """
        keyword_scores = self.extract_keywords(document, n_keywords)
        return [keyword for keyword, score in keyword_scores]

if __name__ == "__main__":
    # 테스트 코드
    extractor = YAKEKeywordExtractor()
    test_doc = """
    Machine learning is a subset of artificial intelligence that focuses on algorithms 
    that can learn from data. Deep learning uses neural networks with multiple layers 
    to model complex patterns in large datasets.
    """
    
    keywords = extractor.extract_keywords(test_doc, 5)
    print("YAKE Keywords:")
    for keyword, score in keywords:
        print(f"  {keyword}: {score:.4f}")
