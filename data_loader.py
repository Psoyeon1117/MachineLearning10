import pandas as pd
import random
import re
from typing import List, Tuple, Optional

class DataLoader:
    def __init__(self, csv_path: str = './clean_data.csv'):
        """
        CSV 데이터 로더 초기화
        
        Args:
            csv_path: CSV 파일 경로
        """
        self.csv_path = csv_path
        self.data = None
        self._load_data()
    
    def _load_data(self):
        """CSV 데이터 로드"""
        try:
            self.data = pd.read_csv(self.csv_path)
            print(f"데이터 로드 완료: {len(self.data)} 문서")
        except Exception as e:
            print(f"데이터 로드 실패: {e}")
            raise
    
    def get_document_by_index(self, index: int) -> Optional[str]:
        """
        인덱스로 문서 가져오기
        
        Args:
            index: 문서 인덱스
            
        Returns:
            문서 텍스트 또는 None
        """
        if self.data is None or index >= len(self.data) or index < 0:
            return None
        return self.data.iloc[index]['Document']
    
    def get_random_documents(self, n: int = 10) -> List[Tuple[int, str]]:
        """
        랜덤 문서 n개 가져오기
        
        Args:
            n: 가져올 문서 수
            
        Returns:
            (인덱스, 문서텍스트) 튜플 리스트
        """
        if self.data is None:
            return []
        
        indices = random.sample(range(len(self.data)), min(n, len(self.data)))
        return [(idx, self.data.iloc[idx]['Document']) for idx in indices]
    
    def preprocess_text(self, text: str) -> str:
        """
        텍스트 전처리
        
        Args:
            text: 원본 텍스트
            
        Returns:
            전처리된 텍스트
        """
        if not isinstance(text, str):
            return ""
        
        # 특수 문자 및 불필요한 공백 정리
        text = re.sub(r'\s+', ' ', text.strip())
        return text
    
    def get_total_documents(self) -> int:
        """전체 문서 수 반환"""
        return len(self.data) if self.data is not None else 0
