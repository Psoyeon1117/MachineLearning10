#!/usr/bin/env python3
"""
모델별 성능 비교 및 분석 스크립트
"""

import time
import statistics
from typing import Dict, List, Tuple
from data_loader import DataLoader
from tfidf_extractor import TFIDFKeywordExtractor
from yake_extractor import YAKEKeywordExtractor
from rake_extractor import RAKEKeywordExtractor
from keybert_extractor import KeyBERTKeywordExtractor

class ModelComparison:
    def __init__(self, csv_path: str = './clean_data.csv'):
        """모델 비교 클래스 초기화"""
        self.data_loader = DataLoader(csv_path)
        self.extractors = {
            'TF-IDF': TFIDFKeywordExtractor(),
            'YAKE': YAKEKeywordExtractor(),
            'RAKE': RAKEKeywordExtractor(),
            'KeyBERT': KeyBERTKeywordExtractor()
        }
    
    def benchmark_models(self, n_docs: int = 20, n_keywords: int = 5) -> Dict:
        """
        모델들의 성능 벤치마크
        
        Args:
            n_docs: 테스트할 문서 수
            n_keywords: 추출할 키워드 수
            
        Returns:
            벤치마크 결과 딕셔너리
        """
        print(f"모델 성능 비교 시작 (문서 {n_docs}개)")
        print("=" * 60)
        
        # 랜덤 문서 가져오기
        test_docs = self.data_loader.get_random_documents(n_docs)
        
        if not test_docs:
            print("테스트 문서를 가져올 수 없습니다.")
            return {}
        
        results = {}
        
        for model_name, extractor in self.extractors.items():
            print(f"\n[{model_name}] 테스트 중...")
            
            times = []
            successful_extractions = 0
            total_keywords_extracted = 0
            keyword_lengths = []
            
            for i, (doc_idx, document) in enumerate(test_docs):
                print(f"  문서 {i+1}/{n_docs} 처리 중...", end='\r')
                
                start_time = time.time()
                
                try:
                    keywords = extractor.get_keywords_only(document, n_keywords)
                    end_time = time.time()
                    
                    processing_time = end_time - start_time
                    times.append(processing_time)
                    
                    if keywords:
                        successful_extractions += 1
                        total_keywords_extracted += len(keywords)
                        keyword_lengths.extend([len(kw) for kw in keywords])
                    
                except Exception as e:
                    end_time = time.time()
                    times.append(end_time - start_time)
                    print(f"  오류 발생 (문서 {doc_idx}): {e}")
            
            # 통계 계산
            avg_time = statistics.mean(times) if times else 0
            median_time = statistics.median(times) if times else 0
            success_rate = successful_extractions / len(test_docs) * 100
            avg_keywords_per_doc = total_keywords_extracted / successful_extractions if successful_extractions > 0 else 0
            avg_keyword_length = statistics.mean(keyword_lengths) if keyword_lengths else 0
            
            results[model_name] = {
                'avg_time': avg_time,
                'median_time': median_time,
                'success_rate': success_rate,
                'total_time': sum(times),
                'avg_keywords_per_doc': avg_keywords_per_doc,
                'avg_keyword_length': avg_keyword_length,
                'successful_extractions': successful_extractions
            }
            
            print(f"  {model_name} 완료" + " " * 20)
        
        return results
    
    def print_comparison_report(self, results: Dict):
        """비교 결과 리포트 출력"""
        print("\n" + "=" * 80)
        print("모델 성능 비교 리포트")
        print("=" * 80)
        
        # 헤더
        print(f"{'모델':<10} {'평균시간':<10} {'중간시간':<10} {'성공률':<8} {'키워드/문서':<12} {'키워드길이':<10}")
        print("-" * 80)
        
        # 각 모델 결과
        for model_name, stats in results.items():
            print(f"{model_name:<10} "
                  f"{stats['avg_time']:<10.3f} "
                  f"{stats['median_time']:<10.3f} "
                  f"{stats['success_rate']:<8.1f}% "
                  f"{stats['avg_keywords_per_doc']:<12.1f} "
                  f"{stats['avg_keyword_length']:<10.1f}")
        
        print("\n" + "=" * 80)
        print("성능 순위")
        print("=" * 80)
        
        # 속도 순위
        speed_ranking = sorted(results.items(), key=lambda x: x[1]['avg_time'])
        print("\n속도 순위 (빠른 순):")
        for i, (model_name, stats) in enumerate(speed_ranking, 1):
            print(f"  {i}. {model_name}: {stats['avg_time']:.3f}초")
        
        # 성공률 순위
        success_ranking = sorted(results.items(), key=lambda x: x[1]['success_rate'], reverse=True)
        print("\n성공률 순위 (높은 순):")
        for i, (model_name, stats) in enumerate(success_ranking, 1):
            print(f"  {i}. {model_name}: {stats['success_rate']:.1f}%")
    
    def detailed_keyword_analysis(self, n_docs: int = 5):
        """상세 키워드 분석"""
        print("\n" + "=" * 80)
        print("상세 키워드 분석")
        print("=" * 80)
        
        test_docs = self.data_loader.get_random_documents(n_docs)
        
        for i, (doc_idx, document) in enumerate(test_docs):
            print(f"\n문서 {i+1} (인덱스: {doc_idx})")
            print(f"내용: {document[:150]}...")
            print("-" * 60)
            
            for model_name, extractor in self.extractors.items():
                try:
                    keywords_with_scores = extractor.extract_keywords(document, 5)
                    if hasattr(extractor, 'extract_keywords'):
                        print(f"\n{model_name}:")
                        for j, (keyword, score) in enumerate(keywords_with_scores, 1):
                            print(f"  {j}. {keyword} (점수: {score:.4f})")
                    else:
                        keywords = extractor.get_keywords_only(document, 5)
                        print(f"\n{model_name}: {keywords}")
                        
                except Exception as e:
                    print(f"\n{model_name}: 오류 - {e}")

def main():
    """메인 함수"""
    print("키워드 추출 모델 성능 비교 시작")
    
    comparator = ModelComparison()
    
    # 벤치마크 실행
    results = comparator.benchmark_models(n_docs=20, n_keywords=5)
    
    if results:
        # 결과 리포트 출력
        comparator.print_comparison_report(results)
        
        # 상세 분석
        comparator.detailed_keyword_analysis(n_docs=3)
    else:
        print("벤치마크 실행에 실패했습니다.")

if __name__ == "__main__":
    main()
