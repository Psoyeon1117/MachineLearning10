import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from typing import List, Tuple
import string

class TFIDFKeywordExtractor:
    def __init__(self):
        """TF-IDF 키워드 추출기 초기화"""
        self.vectorizer = None
        self._download_nltk_data()
        
    def _download_nltk_data(self):
        """필요한 NLTK 데이터 다운로드"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            try:
                nltk.download('punkt', quiet=True)
            except:
                pass
        
        # punkt_tab 다운로드 시도
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            try:
                nltk.download('punkt_tab', quiet=True)
            except:
                pass
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            try:
                nltk.download('stopwords', quiet=True)
            except:
                pass
    
    def _preprocess_text(self, text: str) -> str:
        """텍스트 전처리"""
        if not isinstance(text, str):
            return ""
        
        # 소문자 변환
        text = text.lower()
        
        # 구두점 제거
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # 숫자 제거
        text = re.sub(r'\d+', '', text)
        
        # 다중 공백을 단일 공백으로
        text = re.sub(r'\s+', ' ', text.strip())
        
        return text
    
    def extract_keywords(self, document: str, n_keywords: int = 5) -> List[Tuple[str, float]]:
        """
        단일 문서에서 키워드 추출
        
        Args:
            document: 입력 문서
            n_keywords: 추출할 키워드 수
            
        Returns:
            (키워드, 점수) 튜플 리스트
        """
        # 전처리
        processed_doc = self._preprocess_text(document)
        
        if not processed_doc.strip():
            return []
        
        # TF-IDF 벡터라이저 설정
        # 영어 불용어와 NLTK 불용어 결합
        try:
            nltk_stopwords = set(stopwords.words('english'))
            all_stopwords = ENGLISH_STOP_WORDS.union(nltk_stopwords)
        except:
            all_stopwords = ENGLISH_STOP_WORDS
        
        # 단일 문서이므로 min_df=1, max_df=1.0으로 설정
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words=list(all_stopwords),
            ngram_range=(1, 2),  # unigram과 bigram
            min_df=1,
            max_df=1.0  # 단일 문서에서는 1.0으로 설정
        )
        
        try:
            # 단일 문서에 대해 TF-IDF 계산
            tfidf_matrix = vectorizer.fit_transform([processed_doc])
            feature_names = vectorizer.get_feature_names_out()
            
            # TF-IDF 점수 가져오기
            tfidf_scores = tfidf_matrix.toarray()[0]
            
            # 점수와 함께 키워드 정렬
            keyword_scores = [(feature_names[i], tfidf_scores[i]) 
                            for i in range(len(feature_names)) if tfidf_scores[i] > 0]
            
            # 점수 기준으로 내림차순 정렬
            keyword_scores.sort(key=lambda x: x[1], reverse=True)
            
            return keyword_scores[:n_keywords]
            
        except Exception as e:
            print(f"TF-IDF 키워드 추출 중 오류: {e}")
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
    extractor = TFIDFKeywordExtractor()
    test_doc = """
    Machine learning is a subset of artificial intelligence that focuses on algorithms 
    that can learn from data. Deep learning uses neural networks with multiple layers 
    to model complex patterns in large datasets.
    """
    
    keywords = extractor.extract_keywords(test_doc, 5)
    print("TF-IDF Keywords:")
    for keyword, score in keywords:
        print(f"  {keyword}: {score:.4f}")
