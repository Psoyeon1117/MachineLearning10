from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from typing import List, Tuple
import re
from collections import Counter

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
            sentence_model = SentenceTransformer(self.model_name)
            self.kw_model = KeyBERT(model=sentence_model)
            print("KeyBERT 모델 로딩 완료")
            
        except Exception as e:
            print(f"KeyBERT 모델 초기화 중 오류: {e}")
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
        
        text = re.sub(r'\s+', ' ', text.strip())
        return text
    
    def _is_good_keyword(self, keyword: str) -> bool:
        """좋은 키워드인지 판단"""
        keyword = keyword.lower().strip()
        
        # 최소 길이
        if len(keyword) < 3:
            return False
        
        # 숫자만 있으면 안됨
        if keyword.isdigit():
            return False
        
        # 명백한 불용어
        bad_words = {
            'say', 'said', 'told', 'tell', 'huffpost', 'post', 'draw', 
            'according', 'former', 'also', 'even', 'still', 'just',
            'one', 'two', 'new', 'old', 'time', 'year', 'day'
        }
        
        if keyword in bad_words:
            return False
        
        # 불용어가 포함된 구문
        if any(bad in keyword for bad in bad_words):
            return False
        
        # 사람 이름은 제외 (두 단어가 모두 대문자로 시작)
        words = keyword.split()
        if len(words) == 2 and all(word[0].isupper() and word[1:].islower() for word in words):
            return False
        
        return True
    
    def _extract_simple_keywords(self, document: str, n_keywords: int) -> List[Tuple[str, float]]:
        """간단하고 효과적인 키워드 추출"""
        try:
            # 기본 추출 - 많이 뽑아서 필터링
            keywords = self.kw_model.extract_keywords(
                document,
                keyphrase_ngram_range=(1, 2),  # 1-2 단어
                stop_words='english'
            )
            
            # 상위 50개 정도 가져옴
            candidates = keywords[:50] if len(keywords) >= 50 else keywords
            
            # 품질 필터링
            good_keywords = []
            used_words = set()
            
            for keyword, score in candidates:
                if self._is_good_keyword(keyword):
                    # 중복 단어 체크
                    keyword_words = set(keyword.lower().split())
                    
                    # 이미 사용된 단어와 겹치지 않으면 추가
                    if not keyword_words.intersection(used_words):
                        good_keywords.append((keyword, score))
                        used_words.update(keyword_words)
                        
                        if len(good_keywords) >= n_keywords:
                            break
            
            return good_keywords
            
        except Exception as e:
            print(f"KeyBERT 기본 추출 실패: {e}")
            return []
    
    def _extract_with_mmr(self, document: str, n_keywords: int) -> List[Tuple[str, float]]:
        """MMR을 사용한 다양성 추출"""
        try:
            keywords = self.kw_model.extract_keywords(
                document,
                keyphrase_ngram_range=(1, 2),
                stop_words='english',
                use_mmr=True,
                diversity=0.8
            )
            
            # 필터링
            good_keywords = []
            for keyword, score in keywords[:20]:
                if self._is_good_keyword(keyword):
                    good_keywords.append((keyword, score))
                    if len(good_keywords) >= n_keywords:
                        break
            
            return good_keywords
            
        except Exception as e:
            print(f"KeyBERT MMR 추출 실패: {e}")
            return []
    
    def _get_top_words_from_text(self, document: str, n_keywords: int) -> List[Tuple[str, float]]:
        """텍스트에서 직접 중요 단어 추출 (fallback)"""
        # 간단한 단어 빈도 기반 추출
        words = re.findall(r'\b[a-zA-Z]{4,}\b', document.lower())
        
        # 불용어 제거
        stop_words = {
            'said', 'told', 'huffpost', 'according', 'former', 'also', 'even',
            'still', 'just', 'with', 'from', 'they', 'have', 'this', 'that',
            'were', 'been', 'their', 'would', 'could', 'should', 'about'
        }
        
        filtered_words = [word for word in words if word not in stop_words and len(word) >= 4]
        
        # 빈도 계산
        word_counts = Counter(filtered_words)
        
        # 상위 키워드 반환
        top_words = word_counts.most_common(n_keywords)
        return [(word, float(count)) for word, count in top_words]
    
    def extract_keywords(self, document: str, n_keywords: int = 5) -> List[Tuple[str, float]]:
        """KeyBERT를 사용한 키워드 추출"""
        if self.kw_model is None:
            print("KeyBERT 모델이 초기화되지 않았습니다.")
            # 모델이 없으면 fallback 사용
            return self._get_top_words_from_text(document, n_keywords)
        
        processed_doc = self._preprocess_text(document)
        
        if not processed_doc.strip():
            return []
        
        # 방법 1: MMR 시도
        keywords = self._extract_with_mmr(processed_doc, n_keywords)
        if len(keywords) >= n_keywords:
            return keywords
        
        # 방법 2: 간단한 방법 시도
        keywords = self._extract_simple_keywords(processed_doc, n_keywords)
        if len(keywords) >= n_keywords:
            return keywords
        
        # 방법 3: fallback - 단어 빈도 기반
        print("KeyBERT 방법들이 실패, 단어 빈도 기반으로 대체")
        return self._get_top_words_from_text(processed_doc, n_keywords)
    
    def get_keywords_only(self, document: str, n_keywords: int = 5) -> List[str]:
        """키워드만 반환 (점수 제외)"""
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
