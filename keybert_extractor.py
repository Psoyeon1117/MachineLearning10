from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from typing import List, Tuple
import re

class KeyBERTKeywordExtractor:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        KeyBERT 키워드 추출기 초기화
        
        Args:
            model_name: 사용할 sentence transformer 모델명
        """
        self.model_name = model_name
        self.kw_model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """KeyBERT 모델 초기화"""
        try:
            print(f"KeyBERT 모델 로딩 중: {self.model_name}")
            # CPU에서 사용할 수 있는 경량 모델 사용
            sentence_model = SentenceTransformer(self.model_name)
            self.kw_model = KeyBERT(model=sentence_model)
            print("KeyBERT 모델 로딩 완료")
            
        except Exception as e:
            print(f"KeyBERT 모델 초기화 중 오류: {e}")
            print("기본 모델로 재시도...")
            try:
                self.kw_model = KeyBERT()
                print("기본 KeyBERT 모델 로딩 완료")
            except Exception as e2:
                print(f"기본 모델 로딩도 실패: {e2}")
                self.kw_model = None
    
    def _preprocess_text(self, text: str) -> str:
        """텍스트 전처리"""
        if not isinstance(text, str):
            return ""
        
        # 다중 공백을 단일 공백으로
        text = re.sub(r'\s+', ' ', text.strip())
        
        return text
    
    def extract_keywords(self, document: str, n_keywords: int = 5) -> List[Tuple[str, float]]:
        """
        KeyBERT를 사용한 키워드 추출
        
        Args:
            document: 입력 문서
            n_keywords: 추출할 키워드 수
            
        Returns:
            (키워드, 점수) 튜플 리스트
        """
        if self.kw_model is None:
            print("KeyBERT 모델이 초기화되지 않았습니다.")
            return []
        
        # 전처리
        processed_doc = self._preprocess_text(document)
        
        if not processed_doc.strip():
            return []
        
        try:
            # KeyBERT 키워드 추출 (기본 매개변수만 사용)
            keywords = self.kw_model.extract_keywords(
                processed_doc,
                keyphrase_ngram_range=(1, 2),  # unigram과 bigram
                stop_words='english'
            )
            
            # 상위 n_keywords개만 선택
            keywords = keywords[:n_keywords]
            
            # 결과 필터링
            filtered_keywords = []
            seen_keywords = set()
            
            for keyword, score in keywords:
                clean_keyword = keyword.strip().lower()
                
                # 중복 체크 및 품질 체크
                if (clean_keyword not in seen_keywords and 
                    len(clean_keyword) > 2 and 
                    not clean_keyword.isdigit()):
                    
                    filtered_keywords.append((keyword, score))
                    seen_keywords.add(clean_keyword)
            
            return filtered_keywords
            
        except Exception as e:
            print(f"KeyBERT 키워드 추출 중 오류: {e}")
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
    extractor = KeyBERTKeywordExtractor()
    test_doc = """
    Machine learning is a subset of artificial intelligence that focuses on algorithms 
    that can learn from data. Deep learning uses neural networks with multiple layers 
    to model complex patterns in large datasets.
    """
    
    keywords = extractor.extract_keywords(test_doc, 5)
    print("KeyBERT Keywords:")
    for keyword, score in keywords:
        print(f"  {keyword}: {score:.4f}")
