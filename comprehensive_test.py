#!/usr/bin/env python3
"""
수정된 키워드 추출 모델 테스트 스크립트
모든 오류 수정 후 종합 테스트
"""

def test_all_models():
    """모든 모델 간단 테스트"""
    print("=" * 60)
    print("키워드 추출 모델 종합 테스트")
    print("=" * 60)
    
    test_text = """
    Machine learning is a powerful subset of artificial intelligence that enables computers 
    to learn and improve from experience without being explicitly programmed. Deep learning, 
    a specialized branch of machine learning, uses neural networks with multiple layers 
    to model and understand complex patterns in large datasets. These technologies are 
    revolutionizing fields like natural language processing, computer vision, and robotics.
    """
    
    results = {}
    
    # 1. TF-IDF 테스트
    print("\n1. TF-IDF 테스트:")
    try:
        from tfidf_extractor import TFIDFKeywordExtractor
        extractor = TFIDFKeywordExtractor()
        keywords = extractor.get_keywords_only(test_text, 5)
        results['TF-IDF'] = keywords
        print(f"   ✓ 성공: {keywords}")
    except Exception as e:
        results['TF-IDF'] = []
        print(f"   ✗ 실패: {e}")
    
    # 2. YAKE 테스트
    print("\n2. YAKE 테스트:")
    try:
        from yake_extractor import YAKEKeywordExtractor
        extractor = YAKEKeywordExtractor()
        keywords = extractor.get_keywords_only(test_text, 5)
        results['YAKE'] = keywords
        print(f"   ✓ 성공: {keywords}")
    except Exception as e:
        results['YAKE'] = []
        print(f"   ✗ 실패: {e}")
    
    # 3. RAKE 테스트 (상세)
    print("\n3. RAKE 테스트:")
    try:
        from rake_extractor import RAKEKeywordExtractor
        extractor = RAKEKeywordExtractor()
        
        # RAKE 상태 확인
        print(f"   RAKE 객체 초기화: {extractor.rake is not None}")
        
        keywords = extractor.get_keywords_only(test_text, 5)
        results['RAKE'] = keywords
        
        if keywords:
            print(f"   ✓ 성공: {keywords}")
        else:
            print(f"   ⚠ 키워드 없음 (fallback 확인 필요)")
            
        # 더 자세한 정보
        detailed_keywords = extractor.extract_keywords(test_text, 5)
        if detailed_keywords:
            print(f"   상세 결과: {[(kw, f'{score:.2f}') for kw, score in detailed_keywords[:3]]}")
            
    except Exception as e:
        results['RAKE'] = []
        print(f"   ✗ 실패: {e}")
        import traceback
        traceback.print_exc()
    
    # 4. KeyBERT 테스트
    print("\n4. KeyBERT 테스트:")
    try:
        from keybert_extractor import KeyBERTKeywordExtractor
        extractor = KeyBERTKeywordExtractor()
        keywords = extractor.get_keywords_only(test_text, 5)
        results['KeyBERT'] = keywords
        print(f"   ✓ 성공: {keywords}")
    except Exception as e:
        results['KeyBERT'] = []
        print(f"   ✗ 실패: {e}")
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("테스트 결과 요약:")
    print("=" * 60)
    
    successful_models = 0
    for model_name, keywords in results.items():
        if keywords:
            print(f"✓ {model_name:8}: {len(keywords)}개 키워드 추출 성공")
            successful_models += 1
        else:
            print(f"✗ {model_name:8}: 키워드 추출 실패")
    
    print(f"\n성공률: {successful_models}/4 ({successful_models/4*100:.1f}%)")
    
    if successful_models >= 3:
        print("🎉 대부분의 모델이 정상 작동합니다!")
    elif successful_models >= 2:
        print("⚠️  일부 모델에 문제가 있지만 사용 가능합니다.")
    else:
        print("❌ 많은 모델에 문제가 있습니다. 설정을 확인하세요.")
    
    return results

def compare_results(results):
    """결과 비교"""
    if not any(results.values()):
        print("비교할 결과가 없습니다.")
        return
    
    print("\n" + "=" * 60)
    print("모델별 키워드 비교:")
    print("=" * 60)
    
    # 모든 키워드 수집
    all_keywords = set()
    for keywords in results.values():
        all_keywords.update([kw.lower() for kw in keywords])
    
    print(f"전체 고유 키워드 수: {len(all_keywords)}")
    
    # 공통 키워드 찾기
    if len([r for r in results.values() if r]) > 1:
        working_models = [(name, keywords) for name, keywords in results.items() if keywords]
        
        if len(working_models) >= 2:
            common_keywords = set([kw.lower() for kw in working_models[0][1]])
            for _, keywords in working_models[1:]:
                common_keywords &= set([kw.lower() for kw in keywords])
            
            if common_keywords:
                print(f"공통 키워드: {list(common_keywords)}")
            else:
                print("공통 키워드가 없습니다.")

def quick_data_test():
    """실제 데이터로 빠른 테스트"""
    print("\n" + "=" * 60)
    print("실제 데이터 테스트:")
    print("=" * 60)
    
    try:
        from data_loader import DataLoader
        
        data_loader = DataLoader('./clean_data.csv')
        print(f"데이터 로드 성공: {data_loader.get_total_documents()}개 문서")
        
        # 첫 번째 문서로 테스트
        test_doc = data_loader.get_document_by_index(0)
        if test_doc:
            print(f"테스트 문서: {test_doc[:100]}...")
            
            # TF-IDF만 빠르게 테스트
            try:
                from tfidf_extractor import TFIDFKeywordExtractor
                extractor = TFIDFKeywordExtractor()
                keywords = extractor.get_keywords_only(test_doc, 3)
                print(f"실제 데이터 TF-IDF 결과: {keywords}")
            except Exception as e:
                print(f"실제 데이터 테스트 실패: {e}")
        else:
            print("테스트 문서를 가져올 수 없습니다.")
            
    except Exception as e:
        print(f"데이터 로드 실패: {e}")

def main():
    """메인 함수"""
    results = test_all_models()
    compare_results(results)
    quick_data_test()
    
    print("\n" + "=" * 60)
    print("다음 단계:")
    print("  1. python3 quick_rake_test.py  # RAKE 상세 디버깅")
    print("  2. python3 debug_rake.py      # RAKE 전문 디버깅")  
    print("  3. ./run_test.sh random       # 전체 시스템 테스트")
    print("=" * 60)

if __name__ == "__main__":
    main()
