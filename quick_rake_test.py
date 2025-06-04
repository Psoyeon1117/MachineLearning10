#!/usr/bin/env python3
"""
RAKE만 빠르게 테스트하는 스크립트
"""

def quick_rake_test():
    """빠른 RAKE 테스트"""
    print("=== RAKE 빠른 테스트 ===")
    
    # 1단계: 라이브러리 직접 테스트
    print("\n1. RAKE 라이브러리 직접 테스트:")
    try:
        from rake_nltk import Rake
        
        # 가장 기본적인 설정
        rake = Rake()
        
        test_text = "Machine learning algorithms learn patterns from data"
        rake.extract_keywords_from_text(test_text)
        
        phrases = rake.get_ranked_phrases()
        scores = rake.get_ranked_phrases_with_scores()
        
        print(f"  직접 추출 성공!")
        print(f"  키워드(점수 없음): {phrases[:3]}")
        print(f"  키워드(점수 포함): {[(p, s) for s, p in scores[:3]]}")
        
    except Exception as e:
        print(f"  직접 테스트 실패: {e}")
    
    # 2단계: 우리 추출기 테스트
    print("\n2. 우리 추출기 테스트:")
    try:
        from rake_extractor import RAKEKeywordExtractor
        
        extractor = RAKEKeywordExtractor()
        
        test_texts = [
            "Machine learning algorithms learn patterns from data.",
            "Natural language processing enables computer understanding of human language.",
            "Deep learning neural networks have multiple hidden layers for complex pattern recognition."
        ]
        
        for i, text in enumerate(test_texts, 1):
            print(f"\n  테스트 {i}: {text[:50]}...")
            keywords = extractor.extract_keywords(text, 3)
            if keywords:
                print(f"    성공: {[kw for kw, score in keywords]}")
            else:
                print(f"    실패: 키워드 없음")
                
    except Exception as e:
        print(f"  추출기 테스트 실패: {e}")
        import traceback
        traceback.print_exc()

def test_fallback():
    """Fallback 메서드 테스트"""
    print("\n=== Fallback 메서드 테스트 ===")
    
    try:
        from rake_extractor import RAKEKeywordExtractor
        
        extractor = RAKEKeywordExtractor()
        
        test_text = "Machine learning algorithms learn patterns from large datasets using statistical methods"
        
        print(f"테스트 텍스트: {test_text}")
        
        # Fallback 메서드 직접 호출
        fallback_keywords = extractor._fallback_keyword_extraction(test_text, 5)
        print(f"Fallback 결과: {fallback_keywords}")
        
    except Exception as e:
        print(f"Fallback 테스트 실패: {e}")

if __name__ == "__main__":
    quick_rake_test()
    test_fallback()
