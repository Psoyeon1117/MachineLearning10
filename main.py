"""
키워드 추출 시스템 메인 실행 파일
"""
import sys
import random
from src.keyword_extraction.tester import KeywordExtractionTester
from src.keyword_extraction.benchmark import comprehensive_benchmark

def quick_test(doc_idx: int = None):
    """빠른 테스트 함수"""
    tester = KeywordExtractionTester()
    
    if doc_idx is not None:
        tester.test_specific_document(doc_idx)
    else:
        # 랜덤 문서 테스트
        tester.load_data()
        if tester.df is not None:
            random_idx = random.randint(0, len(tester.df) - 1)
            print(f"랜덤으로 선택된 문서: {random_idx}")
            tester.test_specific_document(random_idx)

def show_help():
    """도움말 출력"""
    print("키워드 추출 모델 테스트 시스템")
    print("\n사용법:")
    print("  python main.py <명령> [옵션]")
    print("\n명령:")
    print("  benchmark [N]     - N개 문서로 전체 벤치마크 실행 (기본값: 10)")
    print("  test <문서번호>   - 특정 문서에 대해 모든 모델 테스트")
    print("  random           - 랜덤 문서 테스트")
    print("  help             - 이 도움말 출력")
    print("\n예시:")
    print("  python main.py benchmark 5")
    print("  python main.py test 100")
    print("  python main.py random")

def main():
    """메인 함수"""
    if len(sys.argv) < 2:
        show_help()
        return
    
    command = sys.argv[1].lower()
    
    try:
        if command == "benchmark":
            n_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 10
            print(f"벤치마크 실행: {n_samples}개 샘플")
            tester, results = comprehensive_benchmark(n_samples)
            
        elif command == "test":
            if len(sys.argv) < 3:
                print("오류: 문서 번호를 입력해주세요.")
                print("사용법: python main.py test <문서번호>")
                return
            
            doc_idx = int(sys.argv[2])
            quick_test(doc_idx)
            
        elif command == "random":
            quick_test()
            
        elif command == "help":
            show_help()
            
        else:
            print(f"알 수 없는 명령어: {command}")
            show_help()
            
    except ValueError as e:
        print(f"입력 오류: {e}")
        show_help()
    except Exception as e:
        print(f"실행 중 오류 발생: {e}")

if __name__ == "__main__":
    main()
