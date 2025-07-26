#!/bin/bash

# EXAONE-4.0-1.2B MLX 파인튜닝 실행 스크립트
# M1 Max 32core GPU, 32GB RAM 환경 최적화

echo "=========================================="
echo "EXAONE-4.0-1.2B MLX Fine-tuning Script"
echo "M1 Max 32core GPU, 32GB RAM 최적화"
echo "=========================================="

# 환경 변수 설정
export MLX_METAL_BUFFER_CACHE_SIZE=2048  # 메탈 버퍼 캐시 크기 (MB)
export PYTORCH_ENABLE_MPS_FALLBACK=1     # MPS fallback 활성화

# Python 경로 확인
echo "Python 버전 확인:"
python3 --version

# 필요한 패키지 설치
echo "필요한 패키지 설치 중..."
pip3 install -r requirements.txt

# MLX 설치 확인
echo "MLX 설치 확인:"
python3 -c "import mlx.core as mx; print(f'MLX version: {mx.__version__}')"

# 데이터셋 존재 확인
if [ ! -f "./dataset/train.jsonl" ]; then
    echo "Error: 훈련 데이터셋이 없습니다. ./dataset/train.jsonl 파일을 확인해주세요."
    exit 1
fi

if [ ! -f "./dataset/valid.jsonl" ]; then
    echo "Warning: 검증 데이터셋이 없습니다. 훈련만 진행됩니다."
fi

# 출력 디렉토리 생성
mkdir -p ./output
mkdir -p ./cache

echo "훈련 시작..."
echo "모델: LGAI-EXAONE/EXAONE-4.0-1.2B"
echo "배치 크기: 4 (메모리 최적화)"
echo "기울기 누적: 8 스텝"
echo "최대 시퀀스 길이: 2048"
echo "에폭 수: 3"
echo "학습률: 2e-5"

# 훈련 실행
python3 train.py \
    --model_name "LGAI-EXAONE/EXAONE-4.0-1.2B" \
    --train_data_path "./dataset/train.jsonl" \
    --valid_data_path "./dataset/valid.jsonl" \
    --batch_size 4 \
    --gradient_accumulation_steps 8 \
    --max_sequence_length 2048 \
    --num_epochs 3 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --use_lora \
    --lora_rank 64 \
    --lora_alpha 16.0 \
    --output_dir "./output" \
    --seed 42

echo "훈련 완료!"
echo "훈련된 모델은 ./output 디렉토리에 저장됩니다."
