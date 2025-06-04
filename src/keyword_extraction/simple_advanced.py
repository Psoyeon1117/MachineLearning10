"""
Advanced 모델 간소화 버전 - 서브워드 문제 해결에 집중
"""

import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import List, Dict
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from .utils import preprocess_text

class SimpleDistilBERTExtractor:
    def __init__(self, model_name: str = "distilbert-base-multilingual-cased"):
        """간소화된 DistilBERT 키워드 추출기"""
        self.device = torch.device('cpu')
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            print(f"Simple DistilBERT 로드 완료: {model_name}")
        except Exception as e:
            print(f"모델 로드 실패: {e}")
            raise
    
    def extract_keywords_simple(self, text: str, top_k: int = 5) -> List[str]:
        """간소화된 키워드 추출 (TF-IDF + BERT 임베딩 검증)"""
        
        # 1단계: TF-IDF로 후보 키워드 추출 (더 많이)
        candidate_keywords = self._get_tfidf_candidates(text, top_k * 3)
        
        if not candidate_keywords:
            return []
        
        # 2단계: BERT 임베딩으로 품질 검증 및 재순위
        verified_keywords = self._verify_with_bert(text, candidate_keywords, top_k)
        
        return verified_keywords
    
    def _get_tfidf_candidates(self, text: str, n_candidates: int) -> List[str]:
        """TF-IDF로 후보 키워드 추출"""
        try:
            # 영어 특화 TF-IDF
            vectorizer = TfidfVectorizer(
                max_features=1000,
                ngram_range=(1, 2),
                stop_words='english',
                lowercase=True,
                token_pattern=r'[a-zA-Z]{3,}',  # 3글자 이상
                min_df=1,
                max_df=0.95
            )
            
            processed_text = preprocess_text(text)
            tfidf_matrix = vectorizer.fit_transform([processed_text])
            feature_names = vectorizer.get_feature_names_out()
            tfidf_scores = tfidf_matrix.toarray()[0]
            
            # 상위 후보들 추출
            top_indices = np.argsort(tfidf_scores)[::-1]
            candidates = []
            
            for idx in top_indices:
                if len(candidates) >= n_candidates:
                    break
                if tfidf_scores[idx] > 0:
                    word = feature_names[idx]
                    if self._is_good_candidate(word):
                        candidates.append(word)
            
            return candidates
            
        except Exception as e:
            print(f"TF-IDF 후보 추출 실패: {e}")
            return []
    
    def _is_good_candidate(self, word: str) -> bool:
        """후보 키워드 기본 검증"""
        word = word.lower().strip()
        
        # 기본 조건
        if len(word) < 3 or not word.isalpha():
            return False
        
        # 간단한 불용어 체크
        common_words = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 
            'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 
            'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 
            'boy', 'did', 'does', 'each', 'end', 'few', 'got', 'let', 'man', 'men',
            'put', 'run', 'say', 'she', 'too', 'use', 'very', 'well', 'were'
        }
        
        return word not in common_words
    
    def _verify_with_bert(self, text: str, candidates: List[str], top_k: int) -> List[str]:
        """BERT 임베딩으로 키워드 품질 검증"""
        if not candidates:
            return []
        
        try:
            # 문서 임베딩 생성
            doc_embedding = self._get_document_embedding(text)
            
            # 각 후보의 중요도 계산
            candidate_scores = []
            
            for candidate in candidates:
                # 후보 키워드의 컨텍스트 임베딩 생성
                keyword_embedding = self._get_keyword_embedding(candidate, text)
                
                if keyword_embedding is not None and doc_embedding is not None:
                    # 문서와의 관련성 점수 계산
                    relevance = np.dot(keyword_embedding, doc_embedding) / (
                        np.linalg.norm(keyword_embedding) * np.linalg.norm(doc_embedding)
                    )
                    
                    # 키워드 자체의 임베딩 크기 (중요도)
                    magnitude = np.linalg.norm(keyword_embedding)
                    
                    # 종합 점수
                    final_score = relevance * 0.6 + (magnitude / 10.0) * 0.4
                    candidate_scores.append((candidate, final_score))
            
            # 점수순 정렬 및 상위 선택
            candidate_scores.sort(key=lambda x: x[1], reverse=True)
            
            return [word for word, score in candidate_scores[:top_k]]
            
        except Exception as e:
            print(f"BERT 검증 실패, TF-IDF 결과 반환: {e}")
            return candidates[:top_k]
    
    def _get_document_embedding(self, text: str) -> np.ndarray:
        """문서 전체 임베딩 생성"""
        try:
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512,
                padding=True
            )
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # [CLS] 토큰의 임베딩 사용 (문서 표현)
                doc_embedding = outputs.last_hidden_state[0][0]  # CLS token
                
            return doc_embedding.cpu().numpy()
            
        except Exception as e:
            print(f"문서 임베딩 생성 실패: {e}")
            return None
    
    def _get_keyword_embedding(self, keyword: str, context: str) -> np.ndarray:
        """키워드의 컨텍스트 임베딩 생성"""
        try:
            # 키워드가 포함된 짧은 컨텍스트 생성
            keyword_context = f"The keyword {keyword} in context: {context[:200]}"
            
            inputs = self.tokenizer(
                keyword_context,
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding=True
            )
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # 평균 임베딩 사용
                keyword_embedding = outputs.last_hidden_state[0].mean(dim=0)
                
            return keyword_embedding.cpu().numpy()
            
        except Exception as e:
            print(f"키워드 임베딩 생성 실패: {e}")
            return None

def extract_keywords_simple_advanced(document: str, top_k: int = 5) -> List[str]:
    """간소화된 Advanced 키워드 추출 함수"""
    try:
        extractor = SimpleDistilBERTExtractor()
        keywords = extractor.extract_keywords_simple(document, top_k)
        return keywords
    except Exception as e:
        print(f"Simple Advanced 키워드 추출 실패: {e}")
        return []

# 테스트 함수
if __name__ == "__main__":
    test_text = "barack obama white house photograph become master image troll use photo"
    keywords = extract_keywords_simple_advanced(test_text, 5)
    print(f"Simple Advanced 키워드: {keywords}")
