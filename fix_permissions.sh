#!/bin/bash

# 실행 권한 설정
echo "실행 권한 설정 중..."
chmod +x setup.sh
chmod +x run_test.sh

echo "오류 수정된 파일들:"
echo "- tfidf_extractor.py: max_df 문제 수정"
echo "- rake_extractor.py: punkt_tab 다운로드 추가"
echo "- keybert_extractor.py: API 매개변수 수정"
echo "- setup.sh: punkt_tab 다운로드 추가"
echo ""
echo "테스트 실행:"
echo "  python3 test_models.py     # 간단한 테스트"
echo "  ./run_test.sh random      # 전체 테스트"
echo ""
echo "먼저 간단한 테스트를 실행해서 문제가 해결되었는지 확인하세요."
