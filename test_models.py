#!/usr/bin/env python3
"""
간단한 키워드 추출 테스트 스크립트
오류 확인 및 디버깅용
"""

def test_tfidf():
    """TF-IDF 테스트"""
    print("TF-IDF 테스트 중...")
    try:
        from tfidf_extractor import TFIDFKeywordExtractor
        extractor = TFIDFKeywordExtractor()
        
        test_text = """
        Machine learning is a subset of artificial intelligence that focuses on algorithms 
        that can learn from data. Deep learning uses neural networks with multiple layers 
        to model complex patterns in large datasets.
        """
        
        keywords = extractor.extract_keywords(test_text, 3)
        print(f"TF-IDF 성공: {[kw for kw, score in keywords]}")
        return True
    except Exception as e:
        print(f"TF-IDF 오류: {e}")
        return False

def test_yake():
    """YAKE 테스트"""
    print("YAKE 테스트 중...")
    try:
        from yake_extractor import YAKEKeywordExtractor
        extractor = YAKEKeywordExtractor()
        
        test_text = """
        Machine learning is a subset of artificial intelligence that focuses on algorithms 
        that can learn from data. Deep learning uses neural networks with multiple layers 
        to model complex patterns in large datasets.
        """
        
        keywords = extractor.extract_keywords(test_text, 3)
        print(f"YAKE 성공: {[kw for kw, score in keywords]}")
        return True
    except Exception as e:
        print(f"YAKE 오류: {e}")
        return False

def test_rake():
    """더 상세한 RAKE 테스트"""
    print("RAKE 테스트 중...")
    try:
        from rake_extractor import RAKEKeywordExtractor
        extractor = RAKEKeywordExtractor()
        
        test_text = """
        Machine learning is a subset of artificial intelligence that focuses on algorithms 
        that can learn from data. Deep learning uses neural networks with multiple layers 
        to model complex patterns in large datasets.
        """
        
        print("\nRAKE 초기화 상태:")
        print(f"  RAKE 객체: {extractor.rake is not None}")
        
        keywords = extractor.extract_keywords(test_text, 5)
        print(f"\nRAKE 추출 결과: {len(keywords)}개")
        
        if keywords:
            print("\nRAKE 키워드 상세:")
            for i, (kw, score) in enumerate(keywords, 1):
                print(f"  {i}. {kw} (점수: {score:.3f})")
        else:
            print("\nRAKE에서 키워드를 추출하지 못했습니다.")
        
        keywords_only = extractor.get_keywords_only(test_text, 5)
        print(f"\nRAKE 성공: {keywords_only}")
        return len(keywords) > 0
        
    except Exception as e:
        print(f"RAKE 오류: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_keybert():
    """KeyBERT 테스트"""
    print("KeyBERT 테스트 중...")
    try:
        from keybert_extractor import KeyBERTKeywordExtractor
        extractor = KeyBERTKeywordExtractor()
        
        test_text = """
        Machine learning is a subset of artificial intelligence that focuses on algorithms 
        that can learn from data. Deep learning uses neural networks with multiple layers 
        to model complex patterns in large datasets.
        """
        
        keywords = extractor.extract_keywords(test_text, 3)
        print(f"KeyBERT 성공: {[kw for kw, score in keywords]}")
        return True
    except Exception as e:
        print(f"KeyBERT 오류: {e}")
        return False

def main():
    """메인 테스트 함수"""
    print("키워드 추출기 간단 테스트")
    print("=" * 50)
    
    results = {}
    results['TF-IDF'] = test_tfidf()
    print()
    results['YAKE'] = test_yake()
    print()
    results['RAKE'] = test_rake()
    print()
    results['KeyBERT'] = test_keybert()
    
    print("\n" + "=" * 50)
    print("테스트 결과 요약:")
    for model, success in results.items():
        status = "✓ 성공" if success else "✗ 실패"
        print(f"  {model}: {status}")
    
    successful_models = [model for model, success in results.items() if success]
    print(f"\n성공한 모델 수: {len(successful_models)}/4")
    
    if len(successful_models) == 4:
        print("모든 모델이 정상 작동합니다!")
    else:
        print("일부 모델에 문제가 있습니다. 위의 오류 메시지를 확인하세요.")

if __name__ == "__main__":
    main()
