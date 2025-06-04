#!/bin/bash

# 키워드 추출 모델 테스트 실행 스크립트

echo "🔍 키워드 추출 모델 테스트 실행"

# 가상환경 활성화 확인
if [[ "$VIRTUAL_ENV" != *".venv"* ]]; then
    echo "⚠️ 가상환경이 활성화되지 않았습니다."
    echo "💡 활성화 방법: source .venv/bin/activate"
    echo "🔄 자동으로 uv run을 사용합니다..."
    UV_RUN="uv run"
else
    UV_RUN=""
fi

# 인수가 없으면 도움말 출력
if [ $# -eq 0 ]; then
    $UV_RUN python main.py help
    exit 0
fi

# 첫 번째 인수에 따라 실행
case $1 in
    "quick")
        echo "🚀 빠른 테스트 실행 (랜덤 문서 1개)"
        $UV_RUN python main.py random
        ;;
    "benchmark")
        SAMPLES=${2:-5}
        echo "📊 벤치마크 테스트 실행 (샘플: $SAMPLES개)"
        $UV_RUN python main.py benchmark $SAMPLES
        ;;
    "test")
        if [ -z "$2" ]; then
            echo "❌ 문서 번호를 입력해주세요."
            echo "사용법: ./run_test.sh test <문서번호>"
            exit 1
        fi
        echo "🎯 문서 $2 테스트 실행"
        $UV_RUN python main.py test $2
        ;;
    "baseline")
        echo "📈 Baseline 모델만 테스트"
        $UV_RUN python -m src.keyword_extraction.baseline_model
        ;;
    "midlevel")
        echo "🧠 Mid-level 모델만 테스트"
        $UV_RUN python -m src.keyword_extraction.midlevel_model
        ;;
    "advanced")
        echo "🚀 Advanced 모델만 테스트"
        $UV_RUN python -m src.keyword_extraction.advanced_model
        ;;
    *)
        echo "❌ 알 수 없는 명령: $1"
        echo "📋 사용 가능한 명령:"
        echo "  quick       - 빠른 테스트 (랜덤 문서)"
        echo "  benchmark [N] - 벤치마크 테스트 (N개 샘플)"
        echo "  test <번호> - 특정 문서 테스트"
        echo "  baseline    - Baseline 모델만 테스트"
        echo "  midlevel    - Mid-level 모델만 테스트"
        echo "  advanced    - Advanced 모델만 테스트"
        ;;
esac
