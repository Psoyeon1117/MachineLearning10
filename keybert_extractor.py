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
        
        # 기본 정리만
        text = re.sub(r'\s+', ' ', text.strip())
        return text
    
    def extract_keywords(self, document: str, n_keywords: int = 5) -> List[Tuple[str, float]]:
        """
        KeyBERT를 사용한 키워드 추출 (다양성 강화)
        
        Args:
            document: 입력 문서
            n_keywords: 추출할 키워드 수
            
        Returns:
            (키워드, 점수) 튜플 리스트
        """
        if self.kw_model is None:
            print("KeyBERT 모델이 초기화되지 않았습니다.")
            return []
        
        processed_doc = self._preprocess_text(document)
        
        if not processed_doc.strip():
            return []
        
        try:
            all_candidates = []
            
            # 방법 1: MMR로 다양성 최대화 (가장 중요)
            try:
                mmr_keywords = self.kw_model.extract_keywords(
                    processed_doc,
                    keyphrase_ngram_range=(1, 2),
                    stop_words='english',
                    use_mmr=True,
                    diversity=0.8,  # 높은 다양성
                    nr_candidates=50  # 많은 후보에서 다양성 선택
                )
                all_candidates.extend(mmr_keywords[:15])  # MMR 결과를 우선시
            except Exception as e:
                print(f"MMR 방법 실패: {e}")
            
            # 방법 2: MaxSum으로 또 다른 다양성 확보
            try:
                maxsum_keywords = self.kw_model.extract_keywords(
                    processed_doc,
                    keyphrase_ngram_range=(1, 2),
                    stop_words='english',
                    use_maxsum=True,
                    nr_candidates=30
                )
                all_candidates.extend(maxsum_keywords[:10])
            except Exception as e:
                print(f"MaxSum 방법 실패: {e}")
            
            # 방법 3: 더 긴 구문 (2-3 gram) - 다른 관점의 키워드
            try:
                long_phrase_keywords = self.kw_model.extract_keywords(
                    processed_doc,
                    keyphrase_ngram_range=(2, 3),
                    stop_words='english'
                )
                all_candidates.extend(long_phrase_keywords[:8])
            except:
                pass
            
            # 중복 제거 후 점수 기준 정렬
            unique_candidates = {}
            for keyword, score in all_candidates:
                if keyword not in unique_candidates:
                    unique_candidates[keyword] = score
                else:
                    # 더 높은 점수로 업데이트
                    unique_candidates[keyword] = max(unique_candidates[keyword], score)
            
            # 점수 기준으로 정렬
            sorted_candidates = sorted(unique_candidates.items(), key=lambda x: x[1], reverse=True)
            
            # 필터링 및 다양성 보장 중복 제거
            final_keywords = []
            used_words = set()
            
            for keyword, score in sorted_candidates:
                # 단어 단위 중복 체크 (더 엄격하게)
                keyword_words = set(keyword.lower().split())
                
                # 이미 사용된 단어와 30% 이상 겹치면 제외
                if used_words:
                    overlap_ratio = len(keyword_words.intersection(used_words)) / len(keyword_words)
                    if overlap_ratio >= 0.3:  # 30% 이상 겹치면 제외
                        continue
                
                final_keywords.append((keyword, score))
                used_words.update(keyword_words)
                
                if len(final_keywords) >= n_keywords:
                    break
            
            return final_keywords
            
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
