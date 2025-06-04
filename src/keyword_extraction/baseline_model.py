"""
Baseline 모델: TF-IDF + Logistic Regression을 이용한 키워드 추출
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple
import re
from .utils import preprocess_text

class TFIDFKeywordExtractor:
    def __init__(self, max_features: int = 5000, ngram_range: Tuple[int, int] = (1, 2)):
        """
        TF-IDF 기반 키워드 추출기
        
        Args:
            max_features: 최대 특성 수
            ngram_range: n-gram 범위
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english',  # 영어 불용어 사용
            lowercase=True,
            token_pattern=r'[a-zA-Z]{3,}'  # 영어 3글자 이상
        )
        self.fitted = False
        
    def fit(self, documents: List[str]):
        """문서 코퍼스로 TF-IDF 벡터라이저 학습"""
        processed_docs = [preprocess_text(doc) for doc in documents]
        self.vectorizer.fit(processed_docs)
        self.fitted = True
        
    def extract_keywords(self, document: str, top_k: int = 5) -> List[str]:
        """단일 문서에서 키워드 추출"""
        if not self.fitted:
            # 단일 문서로 임시 학습
            self.vectorizer.fit([preprocess_text(document)])
        
        processed_doc = preprocess_text(document)
        
        # TF-IDF 벡터 계산
        tfidf_matrix = self.vectorizer.transform([processed_doc])
        
        # 특성 이름 가져오기
        feature_names = self.vectorizer.get_feature_names_out()
        
        # TF-IDF 점수 가져오기
        tfidf_scores = tfidf_matrix.toarray()[0]
        
        # 점수별로 정렬해서 상위 키워드 추출
        top_indices = np.argsort(tfidf_scores)[::-1][:top_k]
        
        keywords = []
        for idx in top_indices:
            if tfidf_scores[idx] > 0:  # 0보다 큰 점수만
                keywords.append(feature_names[idx])
        
        return keywords[:top_k]

class SVMKeywordExtractor:
    def __init__(self, max_features: int = 5000, ngram_range: Tuple[int, int] = (1, 2)):
        """
        SVM 기반 키워드 추출기 (TF-IDF + 문서 유사도)
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english',  # 영어 불용어 사용
            lowercase=True,
            token_pattern=r'[a-zA-Z]{3,}'  # 영어 3글자 이상
        )
        self.fitted = False
        
    def fit(self, documents: List[str]):
        """문서 코퍼스로 학습"""
        processed_docs = [preprocess_text(doc) for doc in documents]
        self.vectorizer.fit(processed_docs)
        self.fitted = True
        
    def extract_keywords(self, document: str, top_k: int = 5) -> List[str]:
        """키워드 추출 (TF-IDF + 통계적 중요도)"""
        if not self.fitted:
            self.vectorizer.fit([preprocess_text(document)])
        
        processed_doc = preprocess_text(document)
        
        # TF-IDF 행렬 계산
        tfidf_matrix = self.vectorizer.transform([processed_doc])
        feature_names = self.vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.toarray()[0]
        
        # 단어 길이와 TF-IDF 점수를 결합한 점수 계산
        combined_scores = []
        for idx, score in enumerate(tfidf_scores):
            if score > 0:
                word = feature_names[idx]
                # 영어 단어 길이 보너스 (너무 짧거나 긴 단어 페널티)
                length_bonus = 1.0
                if 3 <= len(word) <= 8:
                    length_bonus = 1.2
                elif len(word) > 12:
                    length_bonus = 0.8
                
                final_score = score * length_bonus
                combined_scores.append((word, final_score))
        
        # 점수순으로 정렬
        combined_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 상위 키워드 반환
        keywords = [word for word, score in combined_scores[:top_k]]
        return keywords

def extract_keywords_baseline(document: str, method: str = 'tfidf', top_k: int = 5) -> List[str]:
    """
    베이스라인 키워드 추출 함수
    
    Args:
        document: 입력 문서
        method: 'tfidf' 또는 'svm'
        top_k: 추출할 키워드 수
    
    Returns:
        추출된 키워드 리스트
    """
    if method == 'tfidf':
        extractor = TFIDFKeywordExtractor()
    else:  # svm
        extractor = SVMKeywordExtractor()
    
    keywords = extractor.extract_keywords(document, top_k)
    return keywords

# 테스트 함수
if __name__ == "__main__":
    from .utils import load_data, get_random_samples, print_results
    
    # 데이터 로드
    df = load_data()
    if df is not None:
        # 랜덤 샘플 테스트
        samples = get_random_samples(df, 3)
        
        for doc_idx, document in samples:
            # TF-IDF 방법
            keywords_tfidf = extract_keywords_baseline(document, 'tfidf', 5)
            print_results(doc_idx, document, keywords_tfidf, "TF-IDF Baseline")
            
            # SVM 방법
            keywords_svm = extract_keywords_baseline(document, 'svm', 5)
            print_results(doc_idx, document, keywords_svm, "TF-IDF + SVM Baseline")
