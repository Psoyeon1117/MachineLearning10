#!/usr/bin/env python3
"""
키워드 추출 모델 테스트 스크립트
사용법: python keyword_test.py [random|doc_INDEX]
"""

import sys
import time
from data_loader import DataLoader
from tfidf_extractor import TFIDFKeywordExtractor
from yake_extractor import YAKEKeywordExtractor
from rake_extractor import RAKEKeywordExtractor
from keybert_extractor import KeyBERTKeywordExtractor

class KeywordExtractorTester:
    def __init__(self, csv_path: str = './clean_data.csv'):
        """테스터 초기화"""
        self.data_loader = DataLoader(csv_path)
        self.extractors = {
            'TF-IDF': TFIDFKeywordExtractor(),
            'YAKE': YAKEKeywordExtractor(),
            'RAKE': RAKEKeywordExtractor(),
            'KeyBERT': KeyBERTKeywordExtractor()
        }
        
    def test_single_document(self, doc_index: int, n_keywords: int = 5):
        """단일 문서에 대한 키워드 추출 테스트"""
        document = self.data_loader.get_document_by_index(doc_index)
        
        if document is None:
            print(f"문서 인덱스 {doc_index}를 찾을 수 없습니다.")
            return
        
        print(f"\n{'='*80}")
        print(f"문서 인덱스: {doc_index}")
        print(f"{'='*80}")
        print(f"문서 내용 (처음 200자):")
        print(f"{document[:200]}...")
        print(f"\n{'-'*80}")
        
        # 각 모델별 키워드 추출 및 시간 측정
        for model_name, extractor in self.extractors.items():
            print(f"\n[{model_name}]")
            start_time = time.time()
            
            try:
                keywords = extractor.get_keywords_only(document, n_keywords)
                end_time = time.time()
                
                print(f"추출된 키워드: {keywords}")
                print(f"처리 시간: {end_time - start_time:.3f}초")
                
            except Exception as e:
                end_time = time.time()
                print(f"오류 발생: {e}")
                print(f"처리 시간: {end_time - start_time:.3f}초")
        
        print(f"\n{'='*80}")
    
    def test_random_documents(self, n_docs: int = 10, n_keywords: int = 5):
        """랜덤 문서들에 대한 키워드 추출 테스트"""
        random_docs = self.data_loader.get_random_documents(n_docs)
        
        if not random_docs:
            print("랜덤 문서를 가져올 수 없습니다.")
            return
        
        print(f"\n{'='*80}")
        print(f"랜덤 문서 {n_docs}개 테스트")
        print(f"{'='*80}")
        
        # 각 모델별 평균 처리 시간 계산
        model_times = {name: [] for name in self.extractors.keys()}
        model_results = {name: [] for name in self.extractors.keys()}
        
        for i, (doc_idx, document) in enumerate(random_docs):
            print(f"\n[문서 {i+1}/{n_docs}] 인덱스: {doc_idx}")
            print(f"내용 (처음 100자): {document[:100]}...")
            print("-" * 50)
            
            for model_name, extractor in self.extractors.items():
                start_time = time.time()
                
                try:
                    keywords = extractor.get_keywords_only(document, n_keywords)
                    end_time = time.time()
                    processing_time = end_time - start_time
                    
                    model_times[model_name].append(processing_time)
                    model_results[model_name].append(keywords)
                    
                    print(f"{model_name:8}: {keywords} ({processing_time:.3f}초)")
                    
                except Exception as e:
                    end_time = time.time()
                    processing_time = end_time - start_time
                    model_times[model_name].append(processing_time)
                    model_results[model_name].append([])
                    
                    print(f"{model_name:8}: 오류 - {e} ({processing_time:.3f}초)")
        
        # 성능 요약
        print(f"\n{'='*80}")
        print("성능 요약")
        print(f"{'='*80}")
        
        for model_name in self.extractors.keys():
            times = model_times[model_name]
            if times:
                avg_time = sum(times) / len(times)
                successful_extractions = sum(1 for result in model_results[model_name] if result)
                success_rate = successful_extractions / len(model_results[model_name]) * 100
                
                print(f"{model_name:8}: 평균 {avg_time:.3f}초, 성공률 {success_rate:.1f}%")
            else:
                print(f"{model_name:8}: 데이터 없음")

def main():
    """메인 함수"""
    if len(sys.argv) < 2:
        print("사용법: python keyword_test.py [random|doc_INDEX]")
        print("예제: python keyword_test.py random")
        print("예제: python keyword_test.py doc_1000")
        return
    
    command = sys.argv[1].lower()
    tester = KeywordExtractorTester()
    
    if command == 'random':
        # 랜덤 문서 10개 테스트
        tester.test_random_documents(10, 5)
        
    elif command.startswith('doc_'):
        # 특정 문서 테스트
        try:
            doc_index = int(command.split('_')[1])
            tester.test_single_document(doc_index, 5)
        except (IndexError, ValueError):
            print("올바른 문서 인덱스를 입력하세요. 예: doc_1000")
            return
    else:
        print(f"알 수 없는 명령어: {command}")
        print("사용 가능한 명령어: random, doc_INDEX")

if __name__ == "__main__":
    main()
