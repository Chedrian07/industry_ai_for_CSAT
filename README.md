# 선린인터넷고등학교 3-1 김승중 공업일반 프로젝트

## 🚀 주요 성과

- **파인튜닝 효과**: 원본 모델 대비 **정답률 20%p 향상**으로 우수한 성능을 보임
- **GPT-4o mini 성능**: 파인튜닝 모델 다음으로 좋은 성능을 보이며, 원본 모델 대비 **15%p 향상**
- **전체적인 향상**: 모든 모델이 기본 성능을 넘어서는 결과를 보여주며, 파인튜닝의 효과가 명확함
- **실용적 관점**: 파인튜닝 모델이 절반 수준의 정답률을 달성하여 실용성 입증

## 📊 모델 성능 비교 분석

| 모델 | 정답 수 | 정답률 (%) | 총점 | 득점률 (%) | 성능 향상 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 원본 모델 | 6/20 | 30.0% | 14/50 | 28.0% | - |
| **파인튜닝 모델** | **10/20** | **50.0%** | **23/50** | **46.0%** | **+20.0%p** |
| GPT-4o mini | 9/20 | 45.0% | 20/50 | 40.0% | +15.0%p |

### 📈 시각 자료

#### 주요 인사이트
![주요 인사이트](https://raw.githubusercontent.com/Chedrian07/industry_ai_for_CSAT/main/result/1.png)

#### 모델 성능 비교 분석
![모델 성능 비교 분석](https://raw.githubusercontent.com/Chedrian07/industry_ai_for_CSAT/main/result/3.png)

#### EXAONE 모델 파인튜닝 성능 비교
![EXAONE 모델 파인튜닝 성능 비교](https://raw.githubusercontent.com/Chedrian07/industry_ai_for_CSAT/main/result/4.png)

#### 전체 성능 비교
![전체 성능 비교](https://raw.githubusercontent.com/Chedrian07/industry_ai_for_CSAT/main/result/2.png)

## ⚙️ 교육 데이터셋 자동 생성 시스템

Gemini Vision API를 기반으로 이미지에서 텍스트를 추출하고, 이를 구조화하여 교육용 데이터셋을 자동으로 생성하는 파이프라인을 구축했습니다.

![교육 데이터셋 자동 생성 시스템](https://raw.githubusercontent.com/Chedrian07/industry_ai_for_CSAT/main/result/5.png)

### 핵심 기능
- **지능형 답안 처리**: 답안지 자동 감지, 문제-정답 매핑, 해설 추출 및 연결
- **다각도 개념 추출**: 정의/분류, 표/도표 생성, 다양한 질문 생성
- **품질 보증**: JSON 구문 자동 수정, 부분 파싱 복구, 답변 필드 검증

### 처리 결과 및 통계
- **49+** 디렉토리 처리
- **1K+** 이미지 분석
- **85%** 답변 가용률
- **5K+** 데이터 항목 생성

## 🛠️ 개발 환경

- **디바이스**: MacBook (M1 Max 32GB)
- **프레임워크**:
    - MLX 0.27.0
    - MLX-LM 0.26.2
- **파인튜닝 방식**: LoRA (Low-Rank Adaptation)
- **선정 모델**: `LGAI-EXAONE/EXAONE-4.0-1.2B`

## 📚 데이터셋 정보

- **학습 데이터**: 9,455개
- **검증 데이터**: 1,050개
- **총 데이터**: 10,505개
> 데이터셋은 저작권 문제로 비공개합니다.

## 📜 로그

자세한 모델별 성능 측정 로그는 `result/model_log.txt` 파일에서 확인할 수 있습니다.
