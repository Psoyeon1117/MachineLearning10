"""
공통 유틸리티 함수들
"""
import pandas as pd
import numpy as np
import re
import random
from typing import List, Tuple, Dict, Any

def load_data(file_path: str = './clean_data.csv') -> pd.DataFrame:
    """CSV 파일 로드"""
    try:
        df = pd.read_csv(file_path)
        print(f"데이터 로드 완료: {len(df)}개 문서")
        return df
    except Exception as e:
        print(f"데이터 로드 실패: {e}")
        return None

def preprocess_text(text: str) -> str:
    """텍스트 전처리 (영어 데이터용)"""
    if pd.isna(text):
        return ""
    
    # 기본 정리
    text = str(text).strip()
    
    # 영어 데이터에 맞게 수정: 영문, 숫자, 공백, 기본 구두점만 유지
    text = re.sub(r'[^a-zA-Z0-9\s\.,;:!?\-]', ' ', text)
    
    # 다중 공백 제거
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def get_random_samples(df: pd.DataFrame, n: int = 10) -> List[Tuple[int, str]]:
    """랜덤 샘플 추출"""
    if len(df) < n:
        n = len(df)
    
    random_indices = random.sample(range(len(df)), n)
    samples = []
    
    for idx in random_indices:
        doc = df.iloc[idx]['Document']
        samples.append((idx, doc))
    
    return samples

def evaluate_keywords(extracted_keywords: List[str], reference_text: str) -> Dict[str, Any]:
    """키워드 품질 평가 (간단한 휴리스틱)"""
    if not extracted_keywords:
        return {"quality_score": 0, "coverage": 0, "uniqueness": 1}
    
    # 참조 텍스트에서 키워드 출현 빈도 확인
    text_lower = reference_text.lower()
    keyword_hits = sum(1 for kw in extracted_keywords if kw.lower() in text_lower)
    coverage = keyword_hits / len(extracted_keywords)
    
    # 키워드 유니크성 (중복 제거)
    unique_keywords = len(set(extracted_keywords))
    uniqueness = unique_keywords / len(extracted_keywords)
    
    # 간단한 품질 점수
    quality_score = (coverage * 0.7) + (uniqueness * 0.3)
    
    return {
        "quality_score": round(quality_score, 3),
        "coverage": round(coverage, 3),
        "uniqueness": round(uniqueness, 3),
        "keyword_count": len(extracted_keywords)
    }

def print_results(doc_idx: int, document: str, keywords: List[str], model_name: str):
    """결과 출력"""
    print(f"\n{'='*50}")
    print(f"모델: {model_name}")
    print(f"문서 인덱스: {doc_idx}")
    print(f"문서 내용 (첫 200자): {document[:200]}...")
    print(f"추출된 키워드: {', '.join(keywords)}")
    
    # 품질 평가
    eval_result = evaluate_keywords(keywords, document)
    print(f"품질 평가: {eval_result}")
    print(f"{'='*50}")
