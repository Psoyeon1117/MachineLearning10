"""
키워드 추출 모델 테스터 클래스
"""
import pandas as pd
import numpy as np
import time
import random
from typing import List, Dict, Any, Tuple

# 각 모델 import
from .baseline_model import extract_keywords_baseline
from .midlevel_model import extract_keywords_midlevel
from .advanced_model import extract_keywords_advanced
from .utils import load_data, get_random_samples, evaluate_keywords

class KeywordExtractionTester:
    def __init__(self, data_path: str = './clean_data.csv'):
        """키워드 추출 테스터 초기화"""
        self.data_path = data_path
        self.df = None
        self.test_results = []
        
    def load_data(self):
        """데이터 로드"""
        self.df = load_data(self.data_path)
        if self.df is None:
            raise ValueError("데이터 로드 실패")
        print(f"총 {len(self.df)}개 문서 로드 완료")
        
    def test_single_document(self, doc_idx: int, document: str, model_name: str, **kwargs) -> Dict[str, Any]:
        """단일 문서에 대한 키워드 추출 테스트"""
        start_time = time.time()
        
        try:
            # 모델별 키워드 추출
            if model_name.startswith('baseline'):
                method = kwargs.get('method', 'tfidf')
                keywords = extract_keywords_baseline(document, method, 5)
            elif model_name.startswith('midlevel'):
                method = kwargs.get('method', 'bilstm')
                keywords = extract_keywords_midlevel(document, method, 5)
            elif model_name.startswith('advanced'):
                method = kwargs.get('method', 'distilbert')
                approach = kwargs.get('approach', 'clustering')
                keywords = extract_keywords_advanced(document, method, approach, 5)
            else:
                raise ValueError(f"알 수 없는 모델: {model_name}")
            
            # 실행 시간 계산
            execution_time = time.time() - start_time
            
            # 키워드 품질 평가
            quality_metrics = evaluate_keywords(keywords, document)
            
            result = {
                'model_name': model_name,
                'doc_idx': doc_idx,
                'keywords': keywords,
                'execution_time': execution_time,
                'keyword_count': len(keywords),
                'quality_score': quality_metrics.get('quality_score', 0),
                'coverage': quality_metrics.get('coverage', 0),
                'uniqueness': quality_metrics.get('uniqueness', 0),
                'success': True
            }
            
        except Exception as e:
            result = {
                'model_name': model_name,
                'doc_idx': doc_idx,
                'keywords': [],
                'execution_time': time.time() - start_time,
                'keyword_count': 0,
                'quality_score': 0,
                'coverage': 0,
                'uniqueness': 0,
                'success': False,
                'error': str(e)
            }
        
        return result
    
    def test_specific_document(self, doc_idx: int, model_name: str = 'all'):
        """특정 문서에 대한 키워드 추출 테스트"""
        if self.df is None:
            self.load_data()
        
        if doc_idx >= len(self.df):
            print(f"오류: 문서 인덱스 {doc_idx}가 범위를 벗어났습니다. (최대: {len(self.df)-1})")
            return
        
        document = self.df.iloc[doc_idx]['Document']
        
        print(f"\n{'='*50}")
        print(f"문서 {doc_idx} 키워드 추출 테스트")
        print(f"{'='*50}")
        print(f"문서 내용 (첫 200자): {document[:200]}...")
        print(f"문서 길이: {len(document)} 문자")
        
        if model_name == 'all':
            # 모든 모델 테스트
            models = [
                ('TF-IDF', 'baseline', {'method': 'tfidf'}),
                ('SVM', 'baseline', {'method': 'svm'}),
                ('BiLSTM', 'midlevel', {'method': 'bilstm'}),
                ('GRU', 'midlevel', {'method': 'gru'}),
                ('DistilBERT (Clustering)', 'advanced', {'method': 'distilbert', 'approach': 'clustering'}),
                ('DistilBERT (Attention)', 'advanced', {'method': 'distilbert', 'approach': 'attention'}),
                ('KLUE BERT', 'advanced', {'method': 'klue'}),
            ]
            
            for display_name, model_type, params in models:
                print(f"\n--- {display_name} ---")
                result = self.test_single_document(doc_idx, document, model_type, **params)
                
                if result['success']:
                    print(f"키워드: {', '.join(result['keywords'])}")
                    print(f"실행 시간: {result['execution_time']:.2f}초")
                    print(f"품질 점수: {result['quality_score']:.3f}")
                else:
                    print(f"실패: {result.get('error', '알 수 없는 오류')}")
        
        else:
            # 특정 모델만 테스트
            result = self.test_single_document(doc_idx, document, model_name)
            
            if result['success']:
                print(f"\n키워드: {', '.join(result['keywords'])}")
                print(f"실행 시간: {result['execution_time']:.2f}초")
                print(f"품질 점수: {result['quality_score']:.3f}")
            else:
                print(f"\n실패: {result.get('error', '알 수 없는 오류')}")
