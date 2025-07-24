import os
import json
import glob
import time
import re
from pathlib import Path
from datetime import datetime
import google.generativeai as genai
from PIL import Image
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GeminiDatasetCreator:
    def __init__(self):
        """Gemini API를 사용한 데이터셋 생성기 초기화"""
        self.setup_gemini()
        self.base_dir = Path(__file__).parent
        self.images_dir = self.base_dir / "images"
        self.output_dir = self.base_dir / "gemini_responses"
        self.output_dir.mkdir(exist_ok=True)
        
        # 루트의 참조 이미지 로드
        self.reference_image = self.load_reference_image()
        
    def setup_gemini(self):
        """Gemini API 설정"""
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY 환경변수가 설정되지 않았습니다.")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        logger.info("✅ Gemini API 설정 완료")
    
    def load_reference_image(self):
        """루트에 있는 참조 이미지 로드"""
        reference_path = self.images_dir / "page_001.png"
        
        if not reference_path.exists():
            logger.warning(f"⚠️  참조 이미지를 찾을 수 없습니다: {reference_path}")
            return None
        
        try:
            reference_image = Image.open(reference_path)
            logger.info(f"📋 참조 이미지 로드 완료: {reference_path.name}")
            return reference_image
        except Exception as e:
            logger.error(f"❌ 참조 이미지 로드 실패: {str(e)}")
            return None
    
    def get_processing_order(self):
        """처리할 디렉토리들의 순서 결정"""
        processing_order = []
        
        # 1. Chapter 디렉토리들 (1-20 순서로, 각 챕터마다 일반->test 순)
        for i in range(1, 21):
            chapter_dir = f"Chapter_{i}"
            chapter_test_dir = f"Chapter_{i}_test"
            
            if (self.images_dir / chapter_dir).exists():
                processing_order.append(chapter_dir)
            if (self.images_dir / chapter_test_dir).exists():
                processing_order.append(chapter_test_dir)
        
        # 2. CSAT_EXAM 디렉토리들 (연도_월 순서로 정렬)
        csat_pattern = re.compile(r'(\d{2})_(\d{2})_CSAT_EXAM')
        csat_dirs = []
        
        for item in self.images_dir.iterdir():
            if item.is_dir() and csat_pattern.match(item.name):
                match = csat_pattern.match(item.name)
                year = int(match.group(1))
                month = int(match.group(2))
                csat_dirs.append((year, month, item.name))
        
        # 연도, 월 순으로 정렬
        csat_dirs.sort(key=lambda x: (x[0], x[1]))
        processing_order.extend([item[2] for item in csat_dirs])
        
        logger.info(f"📋 처리 순서 ({len(processing_order)}개 디렉토리):")
        for i, dir_name in enumerate(processing_order, 1):
            logger.info(f"  {i:2d}. {dir_name}")
        
        return processing_order
    
    def get_prompt_template(self, directory_type):
        """디렉토리 타입에 따른 프롬프트 템플릿"""
        base_prompt = """
이 이미지들을 분석하여 다음 조건에 따라 JSON 형식으로 데이터를 추출해주세요.

**참조: 첫 번째 이미지는 전체 문서의 구조와 맥락을 이해하기 위한 참조 이미지입니다. 이를 바탕으로 나머지 이미지들의 내용을 분석해주세요.**

**🔍 답안지 확인 및 활용 방법:**
1. **답안지 우선 확인**: 이미지 중에 "정답표", "답안지", "answer key", "해설", "정답과 해설" 등이 포함된 이미지가 있는지 먼저 확인하세요.
2. **답안지 구조 파악**: 답안지에서 문항번호와 정답번호의 매칭표를 정확히 읽어내세요.
3. **해설 내용 확인**: 답안지에 해설이나 풀이 과정이 포함되어 있다면 그 내용도 함께 확인하세요.
4. **문제-정답-해설 매칭**: 각 문제 페이지의 문항과 답안지의 정답, 해설을 정확히 연결하세요.
5. **정답 검증**: 답안지에 명시된 정답이 해당 문제의 선택지와 일치하는지 확인하세요.

**중요 지침:**
1. 한 페이지에 여러 문제나 개념이 있다면 각각을 별도의 JSON 객체로 반환하세요.
2. **표, 분류표, 도표가 있는 경우 반드시 모든 세부 내용을 포함하세요.**
3. **분류 코드, 번호, 기호 등도 정확히 추출하세요.**
4. **예시, 사례, 구체적인 수치나 데이터가 있으면 모두 포함하세요.**
5. **개념 설명 시 관련된 모든 하위 분류, 세부 사항도 함께 설명하세요.**

**문제 유형인 경우 (객관식 문제):**
```json
{
  "id": "문제 고유 ID",
  "chapter_info": {
    "chapter_number": "강의 번호",
    "chapter_title": "강의 제목"
  },
  "problem_type": "문제 유형",
  "context": "문제의 제시문 (표, 글 등)",
  "question": "문제의 발문",
  "stimulus_box": {
    "ㄱ": "보기 ㄱ의 내용",
    "ㄴ": "보기 ㄴ의 내용",
    "ㄷ": "보기 ㄷ의 내용",
    "ㄹ": "보기 ㄹ의 내용"
  },
  "options": {
    "①": "선택지 1번 내용",
    "②": "선택지 2번 내용",
    "③": "선택지 3번 내용",
    "④": "선택지 4번 내용",
    "⑤": "선택지 5번 내용"
  },
  "answer": {
    "correct_option": "③",
    "explanation": "답안지 해설: 제조업체에서 제품의 품질을 일정하게 유지하고 호환성을 보장하기 위해 표준화가 필요하다. 추론 과정: 문제에서 제품 표준화의 의의를 묻고 있으며, 선택지 중 ③번이 표준화의 핵심 목적인 품질 일관성과 호환성을 가장 정확히 설명하고 있다.",
    "answer_available": true
  }
}
```

**답안지가 없는 경우 예시:**
```json
{
  "id": "문제 고유 ID",
  "chapter_info": {
    "chapter_number": "강의 번호",
    "chapter_title": "강의 제목"
  },
  "problem_type": "문제 유형",
  "context": "문제의 제시문 (표, 글 등)",
  "question": "문제의 발문",
  "options": {
    "①": "선택지 1번 내용",
    "②": "선택지 2번 내용",
    "③": "선택지 3번 내용",
    "④": "선택지 4번 내용",
    "⑤": "선택지 5번 내용"
  },
  "answer": {
    "correct_option": "unknown",
    "explanation": "정답이 제공되지 않음",
    "answer_available": false
  }
}
```

**정답 처리 가이드라인:**

**우선순위 1: 답안지에서 정답 및 해설 확인**
1. **답안지가 있고 해설도 포함된 경우**: 
   - 답안지에서 해당 문항번호의 정답을 찾아 정확히 기입
   - 답안지의 해설 내용을 바탕으로 추론 과정을 포함한 설명 작성
   - `"correct_option": "③"` (답안지에 표시된 정확한 번호)
   - `"explanation": "답안지 해설: [답안지의 원본 해설 내용]. 추론 과정: [문제 분석과 정답 도출 과정]"`
   - `"answer_available": true`

2. **답안지가 있지만 해설이 없는 경우**:
   - 답안지에서 정답만 확인하고 문제 내용을 바탕으로 간단한 추론 제공
   - `"correct_option": "③"` (답안지에 표시된 정확한 번호)
   - `"explanation": "답안지 기준 정답: ③. 문제 분석: [문제 내용을 바탕으로 한 간단한 분석]"`
   - `"answer_available": true`

3. **답안지는 있지만 해당 문항이 없는 경우**:
   - `"correct_option": "unknown"`
   - `"explanation": "답안지에 해당 문항번호가 없음"`
   - `"answer_available": false`

**우선순위 2: 문제 페이지에서 정답 확인**
4. **문제 페이지에 정답이 명시된 경우**: 
   - 문제 페이지의 정답을 그대로 사용하고 문제 분석 추가
   - `"explanation": "문제 내 정답: [정답]. 분석: [문제 해결 과정 설명]"`
   - `"answer_available": true`

**우선순위 3: 정답을 찾을 수 없는 경우**
5. **답안지도 없고 문제에도 정답이 없는 경우**: 
   - `"correct_option": "unknown"`
   - `"explanation": "정답이 제공되지 않음"`
   - `"answer_available": false`

**⚠️ 중요: 해설 작성 지침**
- 답안지에 해설이 있는 경우: 해설 내용을 바탕으로 추론 과정을 상세히 설명하세요
- 답안지에 정답만 있는 경우: 문제 내용을 분석하여 정답 도출 과정을 간단히 설명하세요
- 정답이 없는 경우에만 임의 추론을 금지하며, "unknown"으로 처리하세요
- 반드시 답안지나 문제의 정답을 우선 확인한 후 해설을 작성하세요

**개념 설명 유형인 경우 (하나의 개념당 여러 instruction 생성):**
이미지에서 하나의 개념을 발견했을 때, 해당 개념의 내용 특성에 따라 여러 개의 JSON 객체를 생성하세요:

```json
[
  {
    "messages": [
      {"role": "system", "content": "당신은 산업기술 전문가입니다."},
      {"role": "user", "content": "[개념명]이란 무엇인가요?"},
      {"role": "assistant", "content": "정의와 기본 개념 설명"}
    ]
  },
  {
    "messages": [
      {"role": "system", "content": "당신은 산업기술 전문가입니다."},
      {"role": "user", "content": "[개념명]의 분류 체계는 어떻게 구성되어 있나요?"},
      {"role": "assistant", "content": "분류표의 모든 세부 항목과 코드 포함"}
    ]
  },
  {
    "messages": [
      {"role": "system", "content": "당신은 산업기술 전문가입니다."},
      {"role": "user", "content": "[개념명]의 구체적인 예시를 들어주세요."},
      {"role": "assistant", "content": "실제 사례와 예시들"}
    ]
  }
]
```

**내용 특성별 질문 생성 가이드:**

📋 **정의/개념이 있는 경우:**
- "[개념명]이란 무엇인가요?"
- "[개념명]의 의미와 특징을 설명해주세요."

📊 **분류표/체계가 있는 경우:**
- "[개념명]의 분류 체계는 어떻게 구성되어 있나요?"
- "[개념명]의 각 분류별 특징을 설명해주세요."
- "KSIC 대분류 A~U는 각각 무엇을 의미하나요?"

📈 **도표/그래프가 있는 경우:**
- "[주제]의 변화 추이는 어떻게 나타나나요?"
- "그래프에서 보여주는 [주제]의 특징은 무엇인가요?"

🔍 **예시/사례가 있는 경우:**
- "[개념명]의 구체적인 예시를 들어주세요."
- "[개념명]이 실제 산업에서 어떻게 적용되나요?"

⚖️ **비교표/장단점이 있는 경우:**
- "A와 B의 차이점은 무엇인가요?"
- "[개념명]의 장점과 단점은 무엇인가요?"

🔄 **과정/단계가 있는 경우:**
- "[과정명]의 단계별 절차를 설명해주세요."
- "[개념명]은 어떤 과정을 거쳐 발전했나요?"

🎯 **활용/목적이 있는 경우:**
- "[개념명]의 활용 목적은 무엇인가요?"
- "[개념명]이 왜 중요한가요?"

**좋은 개념 설명 예시 (KSIC에서 여러 instruction 생성):**
```json
[
  {
    "messages": [
      {"role": "system", "content": "당신은 산업기술 전문가입니다."},
      {"role": "user", "content": "한국표준산업분류(KSIC)란 무엇인가요?"},
      {"role": "assistant", "content": "한국표준산업분류(KSIC)는 국내 모든 산업 활동을 생산 단위의 유사성에 따라 체계적으로 분류한 국가 표준입니다. 통계청에서 관리하며, 공공 행정, 통계 작성, 산업 정책 수립 등에 활용됩니다."}
    ]
  },
  {
    "messages": [
      {"role": "system", "content": "당신은 산업기술 전문가입니다."},
      {"role": "user", "content": "KSIC의 분류 체계는 어떻게 구성되어 있나요?"},
      {"role": "assistant", "content": "KSIC는 5단계로 구성됩니다:\n1. 대분류: A~U (21개 분류)\n2. 중분류: 2자리 숫자 (77개 분류)\n3. 소분류: 3자리 숫자 (232개 분류)\n4. 세분류: 4자리 숫자 (495개 분류)\n5. 세세분류: 5자리 숫자 (1,196개 분류)"}
    ]
  },
  {
    "messages": [
      {"role": "system", "content": "당신은 산업기술 전문가입니다."},
      {"role": "user", "content": "KSIC 대분류 A~U는 각각 무엇을 의미하나요?"},
      {"role": "assistant", "content": "KSIC 대분류는 다음과 같습니다:\n- A: 농업, 임업 및 어업\n- B: 광업\n- C: 제조업\n- D: 전기, 가스, 증기 및 공기조절 공급업\n- E: 수도, 하수 및 폐기물 처리, 원료재생업\n- F: 건설업\n- G: 도매 및 소매업\n- H: 운수 및 창고업\n- I: 숙박 및 음식점업\n- J: 정보통신업\n- K: 금융 및 보험업\n- L: 부동산업\n- M: 전문, 과학 및 기술 서비스업\n- N: 사업시설 관리, 사업 지원 및 임대 서비스업\n- O: 공공 행정, 국방 및 사회보장 행정\n- P: 교육 서비스업\n- Q: 보건업 및 사회복지 서비스업\n- R: 예술, 스포츠 및 여가관련 서비스업\n- S: 협회 및 단체, 수리 및 기타 개인 서비스업\n- T: 가구 내 고용활동 및 달리 분류되지 않은 자가소비 생산활동\n- U: 국제 및 외국기관"}
    ]
  },
  {
    "messages": [
      {"role": "system", "content": "당신은 산업기술 전문가입니다."},
      {"role": "user", "content": "KSIC의 활용 목적은 무엇인가요?"},
      {"role": "assistant", "content": "KSIC는 다음과 같은 목적으로 활용됩니다:\n1. 국민계정 작성 및 경제분석\n2. 산업구조 분석 및 정책 수립\n3. 통계 조사의 표준화\n4. 국제 비교 통계 작성\n5. 행정업무의 효율성 제고\n6. 민간 기업의 시장 분석 및 전략 수립 지원"}
    ]
  }
]
```

**세부 추출 지침:**
1. 각 문제/개념은 별도의 JSON 객체로 분리
2. **하나의 개념에서 여러 측면의 질문을 생성** (정의→분류→예시→활용 등)
3. 여러 개의 JSON 객체가 있다면 JSON 배열로 반환
4. 이미지에서 식별되는 강의 번호와 제목을 정확히 추출
5. 답안과 해설을 정확히 매칭
6. 개념 설명의 경우 내용 특성을 파악하여 적절한 질문들을 생성

**개념별 추출 전략:**
- 하나의 개념 발견 시 → 해당 개념의 모든 측면을 다각도로 질문
- 정의만 있어도 → 최소 2-3개의 질문 (정의, 특징, 중요성)
- 분류표가 있으면 → 추가로 분류 체계 관련 질문
- 예시가 있으면 → 추가로 실제 적용 사례 질문

**특별히 놓치지 말아야 할 정보:**
- 분류표의 모든 항목 (예: KSIC A~U 대분류, 중분류 번호 등)
- 도표, 그래프의 수치와 라벨 (축 제목, 범례, 구체적 값)
- 단계별 프로세스나 절차 (화살표로 연결된 과정들)
- 공식, 계산법, 비율
- 구체적인 예시나 사례 (회사명, 제품명 등)
- 연도, 시대 구분, 시기별 특징
- 용어의 정의와 특징, 구성 요소
- 장단점, 문제점, 한계
- 관련 법규나 기준, 정책
- 비교표의 모든 항목과 차이점
- 발전 단계나 변화 과정

**실제 질문 생성 예시:**
"콜린 클라크의 산업 분류" 개념이 있다면:
- "콜린 클라크의 산업 분류란 무엇인가요?"
- "클라크는 산업을 어떻게 1차, 2차, 3차로 구분했나요?"
- "클라크 분류법의 각 산업별 특징은 무엇인가요?"
- "클라크 분류법이 경제 분석에서 중요한 이유는 무엇인가요?"

현재 분석할 이미지들의 내용을 위 지침에 따라 JSON으로 추출해주세요:
"""
        return base_prompt
    
    def load_images_from_directory(self, directory_path):
        """디렉토리에서 모든 PNG 이미지 로드"""
        png_files = sorted(list(directory_path.glob("*.png")))
        images = []
        
        for png_file in png_files:
            try:
                image = Image.open(png_file)
                images.append((png_file.name, image))
            except Exception as e:
                logger.error(f"❌ 이미지 로드 실패 {png_file}: {str(e)}")
        
        logger.info(f"  📸 로드된 이미지: {len(images)}개")
        return images
    
    def process_directory(self, directory_name, images):
        """디렉토리의 모든 이미지를 한 번에 처리"""
        directory_type = "chapter" if "Chapter" in directory_name else "exam"
        
        # EXAM 이미지들의 경우 참조 이미지를 사용하지 않음
        use_reference = self.reference_image and directory_type != "exam"
        reference_info = " + 참조이미지" if use_reference else ""
        logger.info(f"    🔄 처리 중 ({len(images)}개 이미지{reference_info})")
        
        try:
            # 프롬프트와 이미지들 준비
            prompt = self.get_prompt_template(directory_type)
            content = [prompt]
            
            # 참조 이미지 먼저 추가 (Chapter 타입인 경우에만)
            if use_reference:
                content.append(self.reference_image)
            
            # 디렉토리의 모든 이미지들 추가
            for img_name, img in images:
                content.append(img)
            
            # Gemini API 호출
            response = self.model.generate_content(content)
            response_text = response.text.strip()
            
            # 디버깅용 응답 출력
            logger.info(f"[DEBUG]: Gemini 응답 내용 (처음 500자)")
            logger.info(f"[DEBUG]: {response_text[:500]}...")
            if len(response_text) > 500:
                logger.info(f"[DEBUG]: 전체 길이: {len(response_text)} 문자")
            
            # 결과 저장
            result = {
                "directory": directory_name,
                "reference_image": "page_001.png" if use_reference else None,
                "image_files": [img_name for img_name, _ in images],
                "response": response_text,
                "processed_at": datetime.now().isoformat()
            }
            
            logger.info(f"    ✅ 처리 완료")
            
            # API 호출 제한 고려
            time.sleep(3)
            
            return result
            
        except Exception as e:
            logger.error(f"    ❌ 처리 실패: {str(e)}")
            return None
    
    def process_all_directories(self):
        """모든 디렉토리 순서대로 처리"""
        processing_order = self.get_processing_order()
        all_results = []
        
        logger.info(f"\n{'='*70}")
        logger.info(f"🚀 데이터셋 생성 작업 시작")
        logger.info(f"{'='*70}")
        
        total_dirs = len(processing_order)
        
        for i, directory_name in enumerate(processing_order, 1):
            directory_path = self.images_dir / directory_name
            
            logger.info(f"\n📂 [{i:2d}/{total_dirs}] {directory_name} 처리 중...")
            logger.info(f"{'─'*50}")
            
            if not directory_path.exists():
                logger.warning(f"⚠️  디렉토리가 존재하지 않음: {directory_name}")
                continue
            
            # 디렉토리에서 이미지들 로드
            images = self.load_images_from_directory(directory_path)
            
            if not images:
                logger.warning(f"⚠️  이미지가 없음: {directory_name}")
                continue
            
            # 디렉토리 처리 (모든 이미지 한 번에)
            directory_result = self.process_directory(directory_name, images)
            
            if directory_result:
                all_results.append(directory_result)
                
                # 디렉토리별 중간 저장
                self.save_directory_results(directory_name, [directory_result])
                
                logger.info(f"✅ {directory_name} 완료")
            else:
                logger.error(f"❌ {directory_name} 처리 실패")
        
        # 최종 결과 저장
        self.save_final_results(all_results)
        
        return all_results
    
    def clean_json_text(self, text):
        """JSON 텍스트 정리 (간단하고 직접적인 방법)"""
        import re
        
        logger.info(f"[DEBUG]: JSON 텍스트 정리 시작 (원본 길이: {len(text)})")
        
        # 1. 제어 문자 제거 (탭, 개행 등은 유지)
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        logger.info(f"[DEBUG]: 제어 문자 제거 완료")
        
        # 2. 잘못된 따옴표 수정
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace("'", "'").replace("'", "'")
        logger.info(f"[DEBUG]: 따옴표 정규화 완료")
        
        # 3. JSON 구문 오류 수정 (순차적이고 안전한 방법)
        logger.info(f"[DEBUG]: 순차적 JSON 구문 오류 수정 시작")
        
        original_text = text
        fixed_count = 0
        
        # 단계 1: 가장 기본적인 패턴 수정 (한 번에 하나씩)
        basic_fixes = [
            ('      "assistant",\n        "content":', '      "role": "assistant",\n        "content":'),
            ('      "user",\n        "content":', '      "role": "user",\n        "content":'),
            ('      "system",\n        "content":', '      "role": "system",\n        "content":'),
            ('    "assistant",\n    "content":', '    "role": "assistant",\n    "content":'),
            ('    "user",\n    "content":', '    "role": "user",\n    "content":'),
            ('    "system",\n    "content":', '    "role": "system",\n    "content":'),
        ]
        
        for old_pattern, new_pattern in basic_fixes:
            if old_pattern in text:
                text = text.replace(old_pattern, new_pattern)
                fixed_count += 1
                logger.info(f"[DEBUG]: 기본 수정 ({fixed_count}): {old_pattern[:20]}... → {new_pattern[:20]}...")
        
        # 단계 2: 중괄호 직후 패턴 수정
        brace_fixes = [
            ('{\n        "assistant",', '{\n        "role": "assistant",'),
            ('{\n        "user",', '{\n        "role": "user",'),
            ('{\n        "system",', '{\n        "role": "system",'),
            ('{\n      "assistant",', '{\n      "role": "assistant",'),
            ('{\n      "user",', '{\n      "role": "user",'),
            ('{\n      "system",', '{\n      "role": "system",'),
        ]
        
        for old_pattern, new_pattern in brace_fixes:
            if old_pattern in text:
                text = text.replace(old_pattern, new_pattern)
                fixed_count += 1
                logger.info(f"[DEBUG]: 중괄호 수정 ({fixed_count}): {old_pattern[:15]}... → {new_pattern[:15]}...")
        
        # 단계 3: 남은 간단한 패턴들
        remaining_fixes = [
            ('"assistant",\n        "content":', '"role": "assistant",\n        "content":'),
            ('"user",\n        "content":', '"role": "user",\n        "content":'),
            ('"system",\n        "content":', '"role": "system",\n        "content":'),
        ]
        
        for old_pattern, new_pattern in remaining_fixes:
            if old_pattern in text and '"role":' not in text[max(0, text.find(old_pattern)-20):text.find(old_pattern)]:
                text = text.replace(old_pattern, new_pattern)
                fixed_count += 1
                logger.info(f"[DEBUG]: 남은 패턴 수정 ({fixed_count}): {old_pattern[:20]}...")
        
        # 단계 4: 혹시 중복된 role 키가 생겼다면 정리
        if '"role": "role":' in text:
            text = text.replace('"role": "role":', '"role":')
            logger.info(f"[DEBUG]: 중복 role 키 정리 완료")
            fixed_count += 1
        
        if '"role": "role": "role":' in text:
            text = text.replace('"role": "role": "role":', '"role":')
            logger.info(f"[DEBUG]: 3중 중복 role 키 정리 완료")
            fixed_count += 1
        
        if text != original_text:
            logger.info(f"[DEBUG]: JSON 구문 오류 수정 완료 ({fixed_count}개 수정, 길이 {len(original_text)} → {len(text)})")
        else:
            logger.info(f"[DEBUG]: JSON 구문 오류 수정 불필요 (이미 올바른 형식)")
        
        # 4. 줄바꿈 문자 정규화
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # 5. 연속된 공백 정리
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        
        logger.info(f"[DEBUG]: JSON 텍스트 정리 완료 (최종 길이: {len(text)})")
        return text
    
    def validate_and_fix_answer_fields(self, parsed_data):
        """답안 필드 검증 및 보완 (답안지 처리 포함)"""
        fixed_count = 0
        logger.info(f"[DEBUG]: 답안 필드 검증 시작 - {len(parsed_data)}개 항목")
        
        for i, item in enumerate(parsed_data):
            # 객관식 문제인 경우 답안 필드 검증
            if isinstance(item, dict) and item.get("problem_type") and "객관식" in str(item.get("problem_type", "")):
                logger.info(f"[DEBUG]: 항목 {i+1} - 객관식 문제 검증 중")
                answer = item.get("answer", {})
                
                # answer 필드가 없거나 비어있는 경우
                if not answer or answer is None:
                    logger.info(f"[DEBUG]: 항목 {i+1} - answer 필드 누락, 기본값 설정")
                    item["answer"] = {
                        "correct_option": "unknown",
                        "explanation": "정답이 제공되지 않음 (답안지 또는 문제에서 확인 필요)",
                        "answer_available": False
                    }
                    fixed_count += 1
                else:
                    # correct_option이 None이거나 빈 값인 경우
                    if not answer.get("correct_option") or answer.get("correct_option") in [None, "", "null", "NULL"]:
                        logger.info(f"[DEBUG]: 항목 {i+1} - correct_option 수정: '{answer.get('correct_option')}' → 'unknown'")
                        answer["correct_option"] = "unknown"
                        if not answer.get("explanation"):
                            answer["explanation"] = "정답이 제공되지 않음"
                        fixed_count += 1
                    
                    # explanation이 None이거나 빈 값인 경우
                    if not answer.get("explanation") or answer.get("explanation") in [None, "", "null", "NULL"]:
                        # correct_option이 있는 경우 답안지에서 가져온 것으로 간주
                        if answer.get("correct_option") and answer.get("correct_option") != "unknown":
                            answer["explanation"] = f"답안지 기준 정답: {answer.get('correct_option')}. 추가 분석이 필요함"
                        else:
                            answer["explanation"] = "정답이 제공되지 않음"
                        fixed_count += 1
                    
                    # answer_available 필드 추가/수정
                    if "answer_available" not in answer:
                        answer["answer_available"] = (
                            answer.get("correct_option") not in [None, "", "unknown", "null", "NULL"] and
                            answer.get("correct_option") is not None
                        )
                    
                    # 답안지 기반 정답인지 확인하여 explanation 개선
                    if (answer.get("correct_option") not in ["unknown", None, "", "null", "NULL"] and
                        len(str(answer.get("explanation", "")).strip()) < 20):  # 매우 짧은 설명인 경우
                        # 설명이 부족한 경우 답안지 해설 참조 요청 메시지 추가
                        old_explanation = answer.get("explanation", "")
                        answer["explanation"] = f"답안지 기준 정답: {answer.get('correct_option')}. {old_explanation} (답안지 해설 참조 필요)".strip()
        
        if fixed_count > 0:
            logger.info(f"[DEBUG]: 🔧 답안 필드 보완 완료: {fixed_count}개 항목 수정됨")
        else:
            logger.info(f"[DEBUG]: ✅ 모든 답안 필드가 정상 상태")
        
        return parsed_data
    
    def parse_gemini_response(self, response_text):
        """Gemini 응답에서 JSON 데이터 추출 및 파싱 (개선된 버전)"""
        original_text = response_text
        
        try:
            logger.info(f"[DEBUG]: JSON 파싱 시작")
            
            # 1단계: 마크다운 코드 블록 제거
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
                logger.info(f"[DEBUG]: ```json 블록에서 JSON 추출")
            elif "```" in response_text:
                # 첫 번째 ``` 이후, 마지막 ``` 이전 내용 추출
                parts = response_text.split("```")
                if len(parts) >= 3:
                    response_text = parts[1].strip()
                    logger.info(f"[DEBUG]: ``` 블록에서 JSON 추출 ({len(parts)}개 부분)")
                else:
                    response_text = response_text.replace("```", "").strip()
                    logger.info(f"[DEBUG]: ``` 기호 제거")
            else:
                logger.info(f"[DEBUG]: 코드 블록 없음, 원본 텍스트 사용")
            
            # 2단계: JSON 텍스트 정리
            cleaned_response = self.clean_json_text(response_text)
            logger.info(f"[DEBUG]: 텍스트 정리 완료 (길이: {len(cleaned_response)})")
            
            # 3단계: JSON 파싱 시도
            logger.info(f"[DEBUG]: JSON 파싱 시도 중...")
            parsed_data = json.loads(cleaned_response)
            
            # 4단계: 리스트가 아니면 리스트로 변환
            if not isinstance(parsed_data, list):
                parsed_data = [parsed_data]
                
            # 5단계: 답안 필드 검증 및 보완
            parsed_data = self.validate_and_fix_answer_fields(parsed_data)
            
            logger.info(f"[DEBUG]: ✅ JSON 파싱 성공: {len(parsed_data)}개 항목")
            return parsed_data
            
        except json.JSONDecodeError as e:
            logger.error(f"[DEBUG]: ❌ JSON 파싱 실패: {str(e)}")
            logger.error(f"[DEBUG]: 실패 위치: line {e.lineno}, column {e.colno}")
            logger.error(f"[DEBUG]: 처리된 텍스트 일부: {cleaned_response[:300]}...")
            
            # 실패한 부분 주변 텍스트 출력
            try:
                lines = cleaned_response.split('\n')
                if e.lineno <= len(lines):
                    logger.error(f"[DEBUG]: 문제 라인 {e.lineno}: {lines[e.lineno-1] if e.lineno > 0 else 'N/A'}")
            except:
                pass
            
            # JSON 파싱 실패 시에도 원본 텍스트 반환 (파싱 시도를 위해)
            logger.info(f"[DEBUG]: 부분 파싱 시도 중...")
            partial_data = self.try_partial_json_parsing(cleaned_response, original_text)
            if partial_data:
                return self.validate_and_fix_answer_fields(partial_data)
            return partial_data
            
        except Exception as e:
            logger.error(f"[DEBUG]: ❌ 예상치 못한 파싱 오류: {str(e)}")
            logger.error(f"[DEBUG]: 원본 응답 일부: {original_text[:300]}...")
            return []
    
    def try_partial_json_parsing(self, cleaned_text, original_text):
        """부분적 JSON 파싱 시도 (강화된 버전)"""
        try:
            logger.info(f"[DEBUG]: 부분 파싱 전략 1 - 배열 래핑 시도")
            # 1. 배열로 감싸서 시도
            if not cleaned_text.strip().startswith('['):
                wrapped_text = f"[{cleaned_text}]"
                try:
                    parsed_data = json.loads(wrapped_text)
                    logger.warning(f"[DEBUG]: ✅ 배열 래핑으로 파싱 성공: {len(parsed_data)}개 항목")
                    return parsed_data
                except Exception as wrap_error:
                    logger.info(f"[DEBUG]: 배열 래핑 실패: {str(wrap_error)}")
                    pass
            
            # 2. 여러 개의 완전한 JSON 객체 추출 시도 (모든 객체 추출)
            logger.info(f"[DEBUG]: 부분 파싱 전략 2 - 모든 완전한 객체 추출 시도")
            objects = []
            brace_count = 0
            start_pos = None
            i = 0
            
            while i < len(cleaned_text):
                char = cleaned_text[i]
                if char == '{':
                    if brace_count == 0:  # 새로운 객체 시작
                        start_pos = i
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0 and start_pos is not None:
                        # 완전한 객체 발견
                        single_object = cleaned_text[start_pos:i+1]
                        logger.info(f"[DEBUG]: 객체 발견 (위치: {start_pos}-{i}, 길이: {len(single_object)})")
                        try:
                            parsed_obj = json.loads(single_object)
                            objects.append(parsed_obj)
                            logger.info(f"[DEBUG]: ✅ 객체 파싱 성공 ({len(objects)}번째)")
                        except Exception as obj_error:
                            logger.error(f"[DEBUG]: 객체 파싱 실패 ({len(objects)+1}번째): {str(obj_error)}")
                        start_pos = None
                i += 1
            
            if objects:
                logger.warning(f"[DEBUG]: ✅ 부분 파싱 성공: {len(objects)}개 항목 추출")
                return objects
            
            # 3. 첫 번째 완전한 JSON 객체만 추출 (기존 방식)
            logger.info(f"[DEBUG]: 부분 파싱 전략 3 - 첫 번째 완전한 객체만 추출 시도")
            brace_count = 0
            start_pos = cleaned_text.find('{')
            if start_pos != -1:
                logger.info(f"[DEBUG]: 첫 번째 '{{' 위치: {start_pos}")
                for i, char in enumerate(cleaned_text[start_pos:], start_pos):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            # 완전한 객체 발견
                            single_object = cleaned_text[start_pos:i+1]
                            logger.info(f"[DEBUG]: 완전한 객체 발견 (길이: {len(single_object)})")
                            try:
                                parsed_data = json.loads(single_object)
                                logger.warning(f"[DEBUG]: ✅ 부분 파싱 성공: 1개 항목 추출")
                                return [parsed_data]
                            except Exception as obj_error:
                                logger.error(f"[DEBUG]: 객체 파싱 실패: {str(obj_error)}")
                                break
            else:
                logger.info(f"[DEBUG]: '{{' 기호를 찾을 수 없음")
            
            # 4. 라인별 객체 추출 시도
            logger.info(f"[DEBUG]: 부분 파싱 전략 4 - 라인별 객체 추출 시도")
            lines = cleaned_text.split('\n')
            objects = []
            current_object = ""
            brace_count = 0
            
            for line_num, line in enumerate(lines):
                current_object += line + '\n'
                for char in line:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                
                if brace_count == 0 and current_object.strip():
                    # 완전한 객체가 될 수 있음
                    obj_text = current_object.strip()
                    if obj_text.startswith('{') and obj_text.endswith('}'):
                        try:
                            parsed_obj = json.loads(obj_text)
                            objects.append(parsed_obj)
                            logger.info(f"[DEBUG]: ✅ 라인별 객체 파싱 성공 ({len(objects)}번째, 라인 {line_num})")
                        except:
                            pass
                    current_object = ""
            
            if objects:
                logger.warning(f"[DEBUG]: ✅ 라인별 파싱 성공: {len(objects)}개 항목 추출")
                return objects
            
            logger.error(f"[DEBUG]: ❌ 모든 파싱 시도 실패")
            return []
            
        except Exception as e:
            logger.error(f"[DEBUG]: ❌ 부분 파싱 중 오류: {str(e)}")
            return []
    
    def save_directory_results(self, directory_name, results):
        """디렉토리별 결과 저장 (깔끔한 JSON 형식 + 원본 응답 백업)"""
        if not results:
            return
        
        # 디렉토리 결과에서 JSON 데이터 추출
        clean_data = []
        raw_responses = []
        
        for result in results:
            response_text = result.get("response", "")
            raw_responses.append({
                "directory": directory_name,
                "image_files": result.get("image_files", []),
                "raw_response": response_text,
                "processed_at": result.get("processed_at")
            })
            
            parsed_items = self.parse_gemini_response(response_text)
            clean_data.extend(parsed_items)
        
        # 1. 파싱된 데이터 저장 (성공한 경우)
        if clean_data:
            output_file = self.output_dir / f"{directory_name}.json"
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(clean_data, f, ensure_ascii=False, indent=2)
                logger.info(f"  💾 파싱 데이터 저장: {output_file.name} ({len(clean_data)}개 항목)")
            except Exception as e:
                logger.error(f"  ❌ 파싱 데이터 저장 실패 {directory_name}: {str(e)}")
        
        # 2. 원본 응답 저장 (항상 저장)
        raw_output_file = self.output_dir / f"{directory_name}_raw_response.json"
        try:
            with open(raw_output_file, 'w', encoding='utf-8') as f:
                json.dump(raw_responses, f, ensure_ascii=False, indent=2)
            logger.info(f"  💾 원본 응답 저장: {raw_output_file.name}")
        except Exception as e:
            logger.error(f"  ❌ 원본 응답 저장 실패 {directory_name}: {str(e)}")
        
        # 파싱된 데이터가 없는 경우에도 알림
        if not clean_data:
            logger.warning(f"  ⚠️  JSON 파싱 실패했지만 원본 응답은 저장됨: {directory_name}")
            logger.info(f"  📝 원본 응답 확인: {raw_output_file.name}")
    
    def analyze_dataset_statistics(self, all_clean_data):
        """데이터셋 통계 분석 (답안지 기반 답안 포함)"""
        stats = {
            "total_items": len(all_clean_data),
            "problems": 0,
            "concepts": 0,
            "problems_with_answers": 0,
            "problems_without_answers": 0,
            "problems_with_answer_key": 0,  # 답안지에서 가져온 답안
            "problems_with_inline_answer": 0,  # 문제에 직접 표시된 답안
            "problems_with_detailed_explanation": 0,  # 상세한 해설이 있는 문제
            "problems_with_basic_explanation": 0,  # 기본적인 해설만 있는 문제
            "answer_availability_rate": 0.0
        }
        
        for item in all_clean_data:
            if isinstance(item, dict):
                # 문제 유형 분류
                if item.get("problem_type") and "객관식" in str(item.get("problem_type", "")):
                    stats["problems"] += 1
                    
                    # 답안 유무 확인
                    answer = item.get("answer", {})
                    if answer.get("answer_available") is True or (
                        answer.get("correct_option") and 
                        answer.get("correct_option") not in ["unknown", "null", "", None]
                    ):
                        stats["problems_with_answers"] += 1
                        
                        # 답안 출처 구분
                        explanation = str(answer.get("explanation", ""))
                        if "답안지" in explanation or "answer key" in explanation.lower():
                            stats["problems_with_answer_key"] += 1
                        else:
                            stats["problems_with_inline_answer"] += 1
                        
                        # 해설의 질 평가
                        if ("추론 과정" in explanation or "분석" in explanation or 
                            "해설:" in explanation or len(explanation) > 50):
                            stats["problems_with_detailed_explanation"] += 1
                        else:
                            stats["problems_with_basic_explanation"] += 1
                    else:
                        stats["problems_without_answers"] += 1
                
                elif item.get("messages"):  # 개념 설명 유형
                    stats["concepts"] += 1
        
        # 답안 가용성 비율 계산
        if stats["problems"] > 0:
            stats["answer_availability_rate"] = (stats["problems_with_answers"] / stats["problems"]) * 100
        
        return stats

    def save_final_results(self, all_results):
        """최종 결과 저장 (깔끔한 JSON 형식)"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 모든 디렉토리 결과에서 JSON 데이터 추출
        all_clean_data = []
        
        for result in all_results:
            response_text = result.get("response", "")
            parsed_items = self.parse_gemini_response(response_text)
            all_clean_data.extend(parsed_items)
        
        # 전체 결과 저장
        final_output = self.output_dir / f"all_dataset_{timestamp}.json"
        
        try:
            with open(final_output, 'w', encoding='utf-8') as f:
                json.dump(all_clean_data, f, ensure_ascii=False, indent=2)
            
            # JSONL 형식으로도 저장 (LLM 파인튜닝용)
            jsonl_output = self.output_dir / f"dataset_{timestamp}.jsonl"
            with open(jsonl_output, 'w', encoding='utf-8') as f:
                for item in all_clean_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            # 데이터셋 통계 분석
            dataset_stats = self.analyze_dataset_statistics(all_clean_data)
            
            # 요약 정보 생성
            parsed_files = [str(f) for f in self.output_dir.glob("Chapter_*.json")] + [str(f) for f in self.output_dir.glob("*_CSAT_EXAM.json")]
            raw_files = [str(f) for f in self.output_dir.glob("*_raw_response.json")]
            
            summary = {
                "total_directories": len(set(r["directory"] for r in all_results)),
                "total_processed": len(all_results),
                "total_items": len(all_clean_data),
                "total_images": sum(len(r["image_files"]) for r in all_results),
                "dataset_statistics": dataset_stats,
                "processing_completed_at": datetime.now().isoformat(),
                "output_files": {
                    "json_file": str(final_output),
                    "jsonl_file": str(jsonl_output),
                    "parsed_data_files": parsed_files,
                    "raw_response_files": raw_files
                }
            }
            
            summary_file = self.output_dir / f"processing_summary_{timestamp}.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            
            logger.info(f"\n{'='*70}")
            logger.info(f"🎉 모든 작업 완료!")
            logger.info(f"{'='*70}")
            logger.info(f"📊 처리 결과:")
            logger.info(f"  - 디렉토리: {summary['total_directories']}개")
            logger.info(f"  - 처리 완료: {summary['total_processed']}개")
            logger.info(f"  - 데이터 항목: {summary['total_items']}개")
            logger.info(f"  - 이미지: {summary['total_images']}개")
            logger.info(f"")
            logger.info(f"📈 데이터셋 구성:")
            logger.info(f"  - 객관식 문제: {dataset_stats['problems']}개")
            logger.info(f"  - 개념 설명: {dataset_stats['concepts']}개")
            logger.info(f"")
            logger.info(f"✅ 답안 가용성:")
            logger.info(f"  - 답안 있음: {dataset_stats['problems_with_answers']}개")
            logger.info(f"    ├─ 답안지 기반: {dataset_stats['problems_with_answer_key']}개")
            logger.info(f"    └─ 문제 내 표시: {dataset_stats['problems_with_inline_answer']}개")
            logger.info(f"  - 답안 없음: {dataset_stats['problems_without_answers']}개")
            logger.info(f"  - 답안 가용률: {dataset_stats['answer_availability_rate']:.1f}%")
            logger.info(f"")
            logger.info(f"📝 해설 품질:")
            logger.info(f"  - 상세 해설: {dataset_stats['problems_with_detailed_explanation']}개")
            logger.info(f"  - 기본 해설: {dataset_stats['problems_with_basic_explanation']}개")
            logger.info(f"")
            logger.info(f"📁 출력 파일:")
            logger.info(f"  - 전체 JSON: {final_output}")
            logger.info(f"  - 전체 JSONL: {jsonl_output}")
            logger.info(f"  - 파싱된 개별 파일: {len(parsed_files)}개")
            logger.info(f"  - 원본 응답 파일: {len(raw_files)}개")
            logger.info(f"  - 처리 요약: {summary_file}")
            
        except Exception as e:
            logger.error(f"❌ 최종 저장 실패: {str(e)}")

def main():
    """메인 함수"""
    try:
        logger.info("🚀 Gemini 데이터셋 생성 시작")
        
        # 환경변수 확인
        if not os.getenv('GEMINI_API_KEY'):
            logger.error("❌ GEMINI_API_KEY 환경변수가 설정되지 않았습니다.")
            logger.info("다음 명령어로 API 키를 설정하세요:")
            logger.info("export GEMINI_API_KEY='your_api_key_here'")
            return
        
        # Gemini 데이터셋 생성기 초기화 및 실행
        creator = GeminiDatasetCreator()
        results = creator.process_all_directories()
        
        logger.info(f"🎊 작업 완료! 총 {len(results)}개의 디렉토리가 처리되었습니다.")
        
    except Exception as e:
        logger.error(f"❌ 작업 실패: {str(e)}")
        raise

if __name__ == "__main__":
    main()
