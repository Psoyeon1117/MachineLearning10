"""
키워드 추출 모델 벤치마크 실행
"""
import pandas as pd
from .tester import KeywordExtractionTester
from .utils import get_random_samples

def run_comprehensive_test(tester: KeywordExtractionTester, n_samples: int = 10):
    """종합 성능 테스트"""
    if tester.df is None:
        tester.load_data()
    
    # 랜덤 샘플 선택
    test_samples = get_random_samples(tester.df, n_samples)
    
    # 테스트할 모델 정의
    models_to_test = [
        ('baseline_tfidf', {'method': 'tfidf'}),
        ('baseline_svm', {'method': 'svm'}),
        ('midlevel_bilstm', {'method': 'bilstm'}),
        ('midlevel_gru', {'method': 'gru'}),
        ('advanced_distilbert_clustering', {'method': 'distilbert', 'approach': 'clustering'}),
        ('advanced_distilbert_attention', {'method': 'distilbert', 'approach': 'attention'}),
        ('advanced_klue', {'method': 'klue', 'approach': 'clustering'}),
    ]
    
    print(f"\n{'='*60}")
    print(f"키워드 추출 모델 종합 성능 테스트 시작")
    print(f"테스트 샘플 수: {n_samples}")
    print(f"{'='*60}")
    
    all_results = []
    
    for doc_idx, document in test_samples:
        print(f"\n문서 {doc_idx} 테스트 중... (길이: {len(document)} 문자)")
        
        for model_name, params in models_to_test:
            print(f"  - {model_name} 테스트 중...")
            result = tester.test_single_document(doc_idx, document, model_name, **params)
            all_results.append(result)
            
            if result['success']:
                print(f"    ✓ 성공: {len(result['keywords'])}개 키워드, {result['execution_time']:.2f}초")
            else:
                print(f"    ✗ 실패: {result.get('error', '알 수 없는 오류')}")
    
    tester.test_results = all_results
    return all_results

def analyze_results(test_results):
    """테스트 결과 분석"""
    if not test_results:
        print("테스트 결과가 없습니다.")
        return
    
    # 결과를 DataFrame으로 변환
    df_results = pd.DataFrame(test_results)
    
    print(f"\n{'='*60}")
    print("모델별 성능 분석 결과")
    print(f"{'='*60}")
    
    # 성공률 분석
    success_rate = df_results.groupby('model_name')['success'].mean()
    print("\n[1] 성공률:")
    for model, rate in success_rate.items():
        print(f"  {model:<35}: {rate*100:5.1f}%")
    
    # 성공한 케이스만 필터링
    df_success = df_results[df_results['success'] == True]
    
    if len(df_success) == 0:
        print("\n성공한 테스트가 없습니다.")
        return
    
    # 평균 성능 지표
    print("\n[2] 평균 성능 지표 (성공 케이스만):")
    metrics = ['execution_time', 'keyword_count', 'quality_score', 'coverage', 'uniqueness']
    
    performance_summary = df_success.groupby('model_name')[metrics].agg(['mean', 'std'])
    
    for metric in metrics:
        print(f"\n  {metric.upper()}:")
        for model in performance_summary.index:
            mean_val = performance_summary.loc[model, (metric, 'mean')]
            std_val = performance_summary.loc[model, (metric, 'std')]
            print(f"    {model:<35}: {mean_val:8.3f} (±{std_val:5.3f})")
    
    # 모델별 랭킹
    print("\n[3] 모델 랭킹 (품질 점수 기준):")
    quality_ranking = df_success.groupby('model_name')['quality_score'].mean().sort_values(ascending=False)
    for i, (model, score) in enumerate(quality_ranking.items(), 1):
        print(f"  {i}. {model:<35}: {score:.3f}")
    
    return df_results

def comprehensive_benchmark(n_samples: int = 10):
    """종합 벤치마크 실행"""
    tester = KeywordExtractionTester()
    
    # 종합 테스트 실행
    results = run_comprehensive_test(tester, n_samples)
    
    # 결과 분석
    df_results = analyze_results(results)
    
    return tester, df_results

if __name__ == "__main__":
    # 기본 벤치마크 실행
    tester, results = comprehensive_benchmark(5)
