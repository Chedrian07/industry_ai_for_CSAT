import os
import json
import time
import argparse
import google.generativeai as genai
from pathlib import Path

# Gemini API 설정 (API 키가 있을 때만)
if "GEMINI_API_KEY" in os.environ:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])

class DatasetGenerator:
    def __init__(self, model_type="pro", debug=False):
        # API 키 확인
        if "GEMINI_API_KEY" not in os.environ:
            raise KeyError("GEMINI_API_KEY")
        
        # 모델 선택 (pro 또는 flash)
        if model_type == "flash":
            self.model = genai.GenerativeModel('gemini-2.5-flash-preview-05-20')
            print("🚀 Gemini 2.5 Flash Thinking 모델 사용")
        else:
            self.model = genai.GenerativeModel('gemini-2.5-pro')
            print("💎 Gemini 2.5 pro 모델 사용")
        
        self.base_path = Path("Industrial_Tech_College_Prep_Workbook")
        self.response_path = Path("gemini_response")
        self.response_path.mkdir(exist_ok=True)
        self.debug = debug
        
        # 업로드된 파일들을 추적하기 위한 딕셔너리
        self.uploaded_files = self.load_uploaded_files()
        
        # 답지 PDF를 저장할 변수 (한 번만 업로드)
        self.answer_sheet_file = None
    
    def load_uploaded_files(self):
        """이전에 업로드된 파일들의 정보를 로드"""
        uploaded_files_path = Path(".gemini_uploaded_files.json")
        if uploaded_files_path.exists():
            with open(uploaded_files_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def save_uploaded_files(self):
        """업로드된 파일들의 정보를 저장"""
        with open(".gemini_uploaded_files.json", 'w', encoding='utf-8') as f:
            json.dump(self.uploaded_files, f, ensure_ascii=False, indent=2)
    
    def upload_pdf_file(self, file_path):
        """PDF 파일을 Gemini에 업로드"""
        file_path_str = str(file_path)
        
        # 이미 업로드된 파일인지 확인
        if file_path_str in self.uploaded_files:
            file_uri = self.uploaded_files[file_path_str]
            try:
                # 파일이 여전히 유효한지 확인
                file = genai.get_file(file_uri.split('/')[-1])
                if file.state.name == "ACTIVE":
                    print(f"기존 업로드된 파일 사용: {file_path.name}")
                    return file
            except:
                # 파일이 더 이상 유효하지 않음
                del self.uploaded_files[file_path_str]
        
        # 새로운 파일 업로드
        print(f"파일 업로드 중: {file_path.name}")
        file = genai.upload_file(file_path)
        
        # 업로드 완료까지 대기
        while file.state.name == "PROCESSING":
            print("업로드 처리 중...")
            time.sleep(2)
            file = genai.get_file(file.name)
        
        if file.state.name == "FAILED":
            raise ValueError(f"파일 업로드 실패: {file_path}")
        
        # 업로드된 파일 정보 저장
        self.uploaded_files[file_path_str] = file.uri
        self.save_uploaded_files()
        
        print(f"파일 업로드 완료: {file_path.name}")
        return file
    
    def upload_answer_sheet_once(self):
        """답지 PDF를 한 번만 업로드 (모든 챕터에서 재사용)"""
        if self.answer_sheet_file is not None:
            return  # 이미 업로드됨
        
        # 첫 번째 챕터에서 답지 PDF 찾기
        for chapter_num in range(1, 21):
            chapter_dir = self.base_path / f"Industrial_Tech_College_Prep_Workbook_chapter_{chapter_num}_pdf"
            if chapter_dir.exists():
                pdf_files = list(chapter_dir.glob("*.pdf"))
                for pdf_file in pdf_files:
                    # 답지 파일인지 확인 (파일명에 특정 키워드가 있다고 가정)
                    if any(keyword in pdf_file.name.lower() for keyword in ["answer", "답", "해설", "solution"]):
                        print(f"\n답지 PDF 발견: {pdf_file.name}")
                        self.answer_sheet_file = self.upload_pdf_file(pdf_file)
                        return
        
        print("경고: 답지 PDF를 찾을 수 없습니다.")
    
    def clean_json_response(self, response_text):
        """Gemini 응답에서 JSON 코드 블록 마커를 제거하고 순수한 JSON만 추출"""
        import re
        
        # 코드 블록 마커 패턴들
        patterns = [
            r'```json\s*',  # ```json 시작
            r'```\s*',      # ``` 시작/끝
            r'````json\s*', # ````json 시작
            r'````\s*'      # ```` 시작/끝
        ]
        
        cleaned_text = response_text.strip()
        
        # 모든 패턴 제거
        for pattern in patterns:
            cleaned_text = re.sub(pattern, '', cleaned_text)
        
        # 앞뒤 공백 제거
        cleaned_text = cleaned_text.strip()
        
        return cleaned_text
    
    def check_existing_files(self, chapter_num):
        """특정 챕터의 데이터셋이 이미 생성되었는지 확인"""
        problem_file = self.response_path / f"chapter_{chapter_num}_problems.json"
        concept_file = self.response_path / f"chapter_{chapter_num}_concepts.json"
        
        both_exist = problem_file.exists() and concept_file.exists()
        
        if both_exist:
            print(f"챕터 {chapter_num}의 데이터셋이 이미 존재합니다. 건너뜁니다.")
            return True
        elif problem_file.exists():
            print(f"챕터 {chapter_num}의 문제풀이 데이터셋만 존재합니다. 개념설명만 생성합니다.")
        elif concept_file.exists():
            print(f"챕터 {chapter_num}의 개념설명 데이터셋만 존재합니다. 문제풀이만 생성합니다.")
        
        return False
    
    def get_problem_solving_prompt(self, chapter_num):
        """문제풀이 데이터셋 생성을 위한 프롬프트"""
        return f"""
업로드된 PDF 파일들을 분석하여 다음 작업을 수행해주세요:

**목표**: 챕터 {chapter_num}의 수능실전 문제들을 JSON 형식으로 추출

**절대 중요**: PDF에서 찾을 수 있는 모든 수능실전 문제를 빠짐없이 추출해야 합니다. 보통 10문제 정도가 있으니 모든 문제가 포함되었는지 확인해주세요.

**보기(stimulus_box) 처리 절대 필수사항**:
⚠️ 이 부분이 가장 중요합니다 ⚠️

1. 문제에서 "다음 중 옳은 것만을 <보기>에서 있는 대로 고른 것은?" 같은 표현이 있으면
2. 반드시 PDF에서 해당 문제의 <보기> 섹션을 찾아서
3. ㄱ, ㄴ, ㄷ, ㄹ, ㅁ 등의 각 보기 내용을 stimulus_box에 포함해야 합니다

**보기 추출 예시**:
- PDF에서 이런 보기가 있다면:
  <보기>
  ㄱ. 중세 산업 사회에서는 길드가 운영되었다.
  ㄴ. 근대 산업 사회에서 공장제 기계공업이 발달했다.
  ㄷ. 현대 산업 사회에서는 산업구조가 고도화되었다.
  ㄹ. 신발공업은 경공업에 해당한다.

- JSON에서는 이렇게 표현해야 합니다:
  "stimulus_box": {{
    "ㄱ": "중세 산업 사회에서는 길드가 운영되었다.",
    "ㄴ": "근대 산업 사회에서 공장제 기계공업이 발달했다.",
    "ㄷ": "현대 산업 사회에서는 산업구조가 고도화되었다.",
    "ㄹ": "신발공업은 경공업에 해당한다."
  }}

**작업 지침**:
1. PDF에서 "수능실전" 또는 문제 풀이 섹션을 모두 찾아주세요
2. 각 문제마다 <보기>가 있는지 꼼꼼히 확인하세요
3. <보기>가 있으면 반드시 ㄱ, ㄴ, ㄷ, ㄹ 내용을 추출하세요
4. 찾은 모든 문제를 다음 JSON 형식으로 변환해주세요

```json
{{
  "id": "문제 번호 [1~10]",
  "chapter_info": {{
    "chapter_number": "{chapter_num}",
    "chapter_title": "챕터 제목을 PDF에서 추출"
  }},
  "question": "문제의 발문",
  "context": "문제의 제시문 (표, 글 등)",
  "stimulus_box": {{
    "ㄱ": "보기 ㄱ의 실제 내용을 여기에",
    "ㄴ": "보기 ㄴ의 실제 내용을 여기에",
    "ㄷ": "보기 ㄷ의 실제 내용을 여기에",
    "ㄹ": "보기 ㄹ의 실제 내용을 여기에"
  }},
  "options": {{
    "①": "선택지 1번 내용",
    "②": "선택지 2번 내용", 
    "③": "선택지 3번 내용",
    "④": "선택지 4번 내용",
    "⑤": "선택지 5번 내용"
  }},
  "answer": {{
    "correct_option": "정답 번호",
    "explanation": "상세 해설 (오답피하기 포함)"
  }}
}}
```

**반드시 확인해야 할 사항들**:
✅ 문제에 "~을 <보기>에서 고른 것은?" 표현이 있으면 → stimulus_box에 ㄱ,ㄴ,ㄷ,ㄹ 내용 필수 포함
✅ 보기가 진짜 없는 문제만 → stimulus_box를 빈 객체 {{}} 로 설정
✅ 선택지가 5개 미만인 경우 → 해당하는 선택지만 포함  
✅ 해설에서 "오답피하기" 키워드가 있는 경우 → 반드시 포함
✅ PDF에서 찾을 수 있는 모든 수능실전 문제 → 빠짐없이 포함 (보통 10문제)

**최종 점검**:
- 각 문제를 다시 한번 확인하여 <보기> 섹션이 있는데 stimulus_box가 비어있으면 안됩니다
- <보기>에서 고른다는 표현이 있는 문제는 100% stimulus_box에 ㄱ,ㄴ,ㄷ,ㄹ 내용이 있어야 합니다
- 보기 내용을 찾지 못했다면 PDF를 다시 꼼꼼히 확인해주세요

응답은 반드시 유효한 JSON 배열 형식으로만 해주세요. 보기 내용이 누락되었다면 절대 안됩니다!
"""

    def get_concept_explanation_prompt(self, chapter_num):
        """개념설명 데이터셋 생성을 위한 프롬프트"""
        return f"""
업로드된 PDF 파일들을 분석하여 다음 작업을 수행해주세요:

**목표**: 챕터 {chapter_num}의 모든 개념과 용어를 Chat 형식 데이터셋으로 변환

**절대 필수 요구사항**: 
- 반드시 정확히 300개 이상의 질문-답변 쌍을 생성해야 합니다
- 300개 미만이면 절대 안됩니다
- 개수가 부족하면 더 세분화하여 300개를 채워주세요

**작업 지침**:
1. PDF에서 개념 설명, 용어 정의, 이론 부분을 모두 찾아주세요
2. 다음 전략을 사용하여 반드시 300개 이상의 질문-답변 쌍을 만들어주세요:

**세분화 전략 (300개 달성을 위해 적극 활용)**:
- **모든 용어 정의**: 본문과 보조단에 설명된 모든 용어에 대해 각각의 정의를 묻는 질문
- **목록 분할**: 하나의 목록에 포함된 모든 항목을 각각 별개의 질문으로 분리
- **세부 특성**: 각 개념의 특성, 장단점, 적용 분야를 개별 질문으로 분리
- **비교 질문**: 유사한 개념들 간의 차이점과 공통점을 여러 관점에서 질문
- **실제 적용**: 각 개념이 실제 어떻게 사용되는지에 대한 질문
- **원리 설명**: 작동 원리, 구조, 메커니즘에 대한 세부 질문
- **분류 체계**: 분류 기준, 종류, 유형에 대한 질문
- **역사적 발전**: 각 개념의 발전 과정, 변화
- **예시와 사례**: 구체적인 예시들을 각각 별개 질문으로
- **상황별 적용**: 다양한 상황에서의 적용 방법

3. 각 질문-답변 쌍은 다음 JSON 형식으로 작성:

```json
{{
  "messages": [
    {{"role": "system", "content": "당신은 대한민국 수능 직업탐구 영역 '공업 일반' 과목에 정통한 전문가입니다."}},
    {{"role": "user", "content": "구체적인 질문"}},
    {{"role": "assistant", "content": "상세하고 정확한 답변"}}
  ]
}}
```

**질문 유형 예시 (300개 달성을 위해 다양하게)**:
- "○○○의 정의는 무엇입니까?"
- "○○○의 특징을 설명해주세요"
- "○○○와 △△△의 차이점은 무엇입니까?"
- "○○○이 사용되는 분야는 어디입니까?"
- "○○○의 작동 원리를 설명해주세요"
- "○○○의 장점과 단점은 무엇입니까?"
- "○○○의 종류에는 무엇이 있나요?"
- "○○○는 언제 사용됩니까?"
- "○○○의 구조는 어떻게 되어 있나요?"
- "○○○가 발전한 과정을 설명해주세요"
- "○○○ 권리의 유효 기간은 얼마입니까?"

**절대 필수 확인사항**:
- 반드시 300개 이상의 질문-답변 쌍을 생성해야 합니다
- 모든 답변은 교사 톤으로 상세하고 정확하게 작성
- 중복되는 내용이라도 다른 관점에서 질문을 만들어 개수를 확보
- 모든 전문 용어와 개념을 빠짐없이 포함
- 300개 미만이면 더 세분화하여 개수를 채우세요

**개수 확인**: 응답 전에 생성한 질문-답변 쌍의 개수를 세어보고, 300개 미만이면 더 추가해주세요.

응답은 반드시 유효한 JSON 배열 형식으로만 해주세요. 300개 이상이 되었는지 꼭 확인하세요.
"""

    def get_kice_exam_prompt(self, exam_name):
        """KICE 시험 데이터셋 생성을 위한 프롬프트"""
        return f"""
업로드된 PDF 파일들을 분석하여 다음 작업을 수행해주세요:

**목표**: {exam_name} 시험의 공업일반 과목 문제들을 JSON 형식으로 추출

**절대 중요**: PDF에서 찾을 수 있는 모든 공업일반 문제를 빠짐없이 추출해야 합니다.

**보기(stimulus_box) 처리 절대 필수사항**:
⚠️ 이 부분이 가장 중요합니다 ⚠️

1. 문제에서 "다음 중 옳은 것만을 <보기>에서 있는 대로 고른 것은?" 같은 표현이 있으면
2. 반드시 PDF에서 해당 문제의 <보기> 섹션을 찾아서
3. ㄱ, ㄴ, ㄷ, ㄹ, ㅁ 등의 각 보기 내용을 stimulus_box에 포함해야 합니다

**보기 추출 예시**:
- PDF에서 이런 보기가 있다면:
  <보기>
  ㄱ. 중세 산업 사회에서는 길드가 운영되었다.
  ㄴ. 근대 산업 사회에서 공장제 기계공업이 발달했다.
  ㄷ. 현대 산업 사회에서는 산업구조가 고도화되었다.
  ㄹ. 신발공업은 경공업에 해당한다.

- JSON에서는 이렇게 표현해야 합니다:
  "stimulus_box": {{
    "ㄱ": "중세 산업 사회에서는 길드가 운영되었다.",
    "ㄴ": "근대 산업 사회에서 공장제 기계공업이 발달했다.",
    "ㄷ": "현대 산업 사회에서는 산업구조가 고도화되었다.",
    "ㄹ": "신발공업은 경공업에 해당한다."
  }}

**작업 지침**:
1. PDF에서 공업일반 과목의 모든 문제를 찾아주세요
2. 각 문제마다 <보기>가 있는지 꼼꼼히 확인하세요
3. <보기>가 있으면 반드시 ㄱ, ㄴ, ㄷ, ㄹ 내용을 추출하세요
4. 시험 정보를 정확히 파악하여 EXAM_NAME에 포함하세요
5. 찾은 모든 문제를 다음 JSON 형식으로 변환해주세요

```json
{{
  "id": "문제 번호",
  "EXAM_NAME": "시험 이름",
  "question": "문제의 발문",
  "context": "문제의 제시문 (표, 글 등)",
  "stimulus_box": {{
    "ㄱ": "보기 ㄱ의 실제 내용을 여기에",
    "ㄴ": "보기 ㄴ의 실제 내용을 여기에",
    "ㄷ": "보기 ㄷ의 실제 내용을 여기에",
    "ㄹ": "보기 ㄹ의 실제 내용을 여기에"
  }},
  "options": {{
    "①": "선택지 1번 내용",
    "②": "선택지 2번 내용", 
    "③": "선택지 3번 내용",
    "④": "선택지 4번 내용",
    "⑤": "선택지 5번 내용"
  }},
  "answer": {{
    "correct_option": "정답 번호",
    "explanation": "상세 해설 (오답피하기 포함)"
  }}
}}
```

**반드시 확인해야 할 사항들**:
✅ **explanation 필드에는 직접 Reasoning LLM이 추론하는 듯한 과정으로 보이는 LLM 추론과정을 포함하여 정답에 맞는 해설 과정을 작성해주세요. '해설:'뒤에 내용을 작성합니다.**"
✅ 문제에 "~을 <보기>에서 고른 것은?" 표현이 있으면 → stimulus_box에 ㄱ,ㄴ,ㄷ,ㄹ 내용 필수 포함
✅ 보기가 진짜 없는 문제만 → stimulus_box를 빈 객체 {{}} 로 설정
✅ 선택지가 5개 미만인 경우 → 해당하는 선택지만 포함  
✅ 해설에서 "오답피하기" 키워드가 있는 경우 → 반드시 포함
✅ PDF에서 찾을 수 있는 모든 공업일반 문제 → 빠짐없이 포함
✅ 시험 정보를 정확히 파악하여 EXAM_NAME에 포함 (예: "2017학년도대학수학능력시험6월모의평가")

**최종 점검**:
- 각 문제를 다시 한번 확인하여 <보기> 섹션이 있는데 stimulus_box가 비어있으면 안됩니다
- <보기>에서 고른다는 표현이 있는 문제는 100% stimulus_box에 ㄱ,ㄴ,ㄷ,ㄹ 내용이 있어야 합니다

응답은 반드시 유효한 JSON 배열 형식으로만 해주세요.
"""

    def process_chapter(self, chapter_num):
        """특정 챕터의 PDF들을 처리"""
        # 이미 생성된 파일이 있는지 확인
        if self.check_existing_files(chapter_num):
            return
        
        chapter_dir = self.base_path / f"Industrial_Tech_College_Prep_Workbook_chapter_{chapter_num}_pdf"
        
        if not chapter_dir.exists():
            print(f"챕터 {chapter_num} 디렉토리가 존재하지 않습니다: {chapter_dir}")
            return
        
        print(f"\n=== 챕터 {chapter_num} 처리 시작 ===")
        
        # 챕터 디렉토리 내의 PDF 파일들 찾기
        pdf_files = list(chapter_dir.glob("*.pdf"))
        
        if not pdf_files:
            print(f"챕터 {chapter_num}에서 PDF 파일을 찾을 수 없습니다.")
            return
        
        print(f"발견된 PDF 파일들: {[f.name for f in pdf_files]}")
        
        # PDF 파일들 업로드 (답지 제외)
        uploaded_files = []
        for pdf_file in pdf_files:
            try:
                # 답지가 아닌 파일만 업로드
                if not any(keyword in pdf_file.name.lower() for keyword in ["answer", "답", "해설", "solution"]):
                    uploaded_file = self.upload_pdf_file(pdf_file)
                    uploaded_files.append(uploaded_file)
            except Exception as e:
                print(f"파일 업로드 실패 {pdf_file.name}: {e}")
                continue
        
        # 답지 파일도 추가 (이미 업로드됨)
        if self.answer_sheet_file:
            uploaded_files.append(self.answer_sheet_file)
        
        if not uploaded_files:
            print(f"챕터 {chapter_num}에서 업로드된 파일이 없습니다.")
            return
        
        # 1. 문제풀이 데이터셋 생성 (이미 존재하지 않는 경우만)
        problem_file = self.response_path / f"chapter_{chapter_num}_problems.json"
        if not problem_file.exists():
            print(f"\n--- 챕터 {chapter_num} 문제풀이 데이터셋 생성 중 ---")
            try:
                problem_prompt = self.get_problem_solving_prompt(chapter_num)
                
                if self.debug:
                    print("🔍 문제풀이 프롬프트 전송 중...")
                    print("=" * 80)
                    print("📝 문제풀이 프롬프트 내용:")
                    print("-" * 80)
                    print(problem_prompt[:1000] + "..." if len(problem_prompt) > 1000 else problem_prompt)
                    print("=" * 80)
                    print("📤 Gemini API 호출 시작...")
                
                problem_response = self.model.generate_content(
                    [problem_prompt] + uploaded_files,
                    stream=True if self.debug else False
                )
                
                if self.debug:
                    print("📥 Gemini 실시간 응답 스트리밍 시작...")
                    print("=" * 80)
                    print("🤖 실시간 Gemini 문제풀이 응답:")
                    print("-" * 80)
                    
                    full_response = ""
                    for chunk in problem_response:
                        if chunk.text:
                            print(chunk.text, end="", flush=True)
                            full_response += chunk.text
                    
                    print("\n" + "=" * 80)
                    print("📥 응답 스트리밍 완료")
                    print("🧹 응답 정리 중...")
                    print(f"📄 전체 응답 길이: {len(full_response)} 문자")
                    
                    # 스트리밍 응답을 단일 응답 객체로 변환
                    class ResponseWrapper:
                        def __init__(self, text):
                            self.text = text
                    
                    problem_response = ResponseWrapper(full_response)
                else:
                    print("📥 Gemini 응답 수신 완료")
                    print("🧹 응답 정리 중...")
                    print(f"📄 원본 응답 길이: {len(problem_response.text)} 문자")
                
                # JSON 응답에서 코드 블록 마커 제거
                cleaned_response = self.clean_json_response(problem_response.text)
                
                if self.debug:
                    print(f"✂️ 정리된 응답 길이: {len(cleaned_response)} 문자")
                    print("🔍 JSON 검증 중...")
                
                # 응답 검증
                try:
                    problems_data = json.loads(cleaned_response)
                    problem_count = len(problems_data)
                    print(f"생성된 문제 수: {problem_count}개")
                    
                    if self.debug:
                        print(f"📊 문제별 세부 정보:")
                        for i, problem in enumerate(problems_data):
                            stimulus_count = len(problem.get('stimulus_box', {}))
                            print(f"  문제 {i+1}: ID={problem.get('id', '?')}, 보기수={stimulus_count}")
                    
                    if problem_count < 8:  # 최소 8문제는 있어야 함
                        print(f"경고: 문제 수가 부족합니다 ({problem_count}개). 더 많은 문제가 있는지 확인이 필요합니다.")
                except json.JSONDecodeError as e:
                    print(f"경고: JSON 파싱 실패. 응답 형식을 확인해주세요. 오류: {e}")
                    if self.debug:
                        print("💥 JSON 파싱 실패한 응답 내용 (처음 500자):")
                        print(cleaned_response[:500])
                
                # 응답 저장
                with open(problem_file, 'w', encoding='utf-8') as f:
                    f.write(cleaned_response)
                
                print(f"문제풀이 데이터셋 저장됨: {problem_file}")
                
            except Exception as e:
                print(f"문제풀이 데이터셋 생성 실패: {e}")
                if self.debug:
                    import traceback
                    print("❌ 상세 오류 정보:")
                    traceback.print_exc()
        
        # 2. 개념설명 데이터셋 생성 (이미 존재하지 않는 경우만)
        concept_file = self.response_path / f"chapter_{chapter_num}_concepts.json"
        if not concept_file.exists():
            print(f"\n--- 챕터 {chapter_num} 개념설명 데이터셋 생성 중 ---")
            try:
                concept_prompt = self.get_concept_explanation_prompt(chapter_num)
                
                if self.debug:
                    print("🔍 개념설명 프롬프트 전송 중...")
                    print("=" * 80)
                    print("📝 개념설명 프롬프트 내용:")
                    print("-" * 80)
                    print(concept_prompt[:1000] + "..." if len(concept_prompt) > 1000 else concept_prompt)
                    print("=" * 80)
                    print("📤 Gemini API 호출 시작 (이 과정은 시간이 걸릴 수 있습니다)...")
                
                concept_response = self.model.generate_content(
                    [concept_prompt] + uploaded_files,
                    stream=True if self.debug else False,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.3,
                        max_output_tokens=64000,
                        top_p=0.8,
                        top_k=40
                    )
                )
                
                if self.debug:
                    print("📥 Gemini 실시간 응답 스트리밍 시작...")
                    print("=" * 80)
                    print("🤖 실시간 Gemini 개념설명 응답:")
                    print("-" * 80)
                    
                    full_response = ""
                    for chunk in concept_response:
                        if chunk.text:
                            print(chunk.text, end="", flush=True)
                            full_response += chunk.text
                    
                    print("\n" + "=" * 80)
                    print("📥 응답 스트리밍 완료")
                    print("🧹 응답 정리 중...")
                    print(f"📄 전체 응답 길이: {len(full_response)} 문자")
                    
                    # 스트리밍 응답을 단일 응답 객체로 변환
                    class ResponseWrapper:
                        def __init__(self, text):
                            self.text = text
                    
                    concept_response = ResponseWrapper(full_response)
                else:
                    print("📥 Gemini 응답 수신 완료")
                    print("🧹 응답 정리 중...")
                    print(f"📄 원본 응답 길이: {len(concept_response.text)} 문자")
                
                # JSON 응답에서 코드 블록 마커 제거
                cleaned_response = self.clean_json_response(concept_response.text)
                
                if self.debug:
                    print(f"✂️ 정리된 응답 길이: {len(cleaned_response)} 문자")
                    print("🔍 JSON 검증 중...")
                
                # 응답 검증
                try:
                    concepts_data = json.loads(cleaned_response)
                    concept_count = len(concepts_data)
                    print(f"생성된 개념 질문-답변 쌍 수: {concept_count}개")
                    
                    if self.debug:
                        print(f"📊 개념 데이터 세부 정보:")
                        for i, concept in enumerate(concepts_data[:3]):  # 처음 3개만 표시
                            user_content = concept.get('messages', [{}])[1].get('content', 'N/A')[:50]
                            print(f"  개념 {i+1}: {user_content}...")
                        if concept_count > 3:
                            print(f"  ... 및 {concept_count-3}개 추가")
                    
                    if concept_count < 300:
                        print(f"경고: 개념 질문-답변 쌍이 300개 미만입니다 ({concept_count}개). 목표는 300개 이상입니다.")
                        
                except json.JSONDecodeError as e:
                    print(f"경고: JSON 파싱 실패. 응답 형식을 확인해주세요. 오류: {e}")
                    if self.debug:
                        print("💥 JSON 파싱 실패한 응답 내용 (처음 500자):")
                        print(cleaned_response[:500])
                
                # 응답 저장
                with open(concept_file, 'w', encoding='utf-8') as f:
                    f.write(cleaned_response)
                
                print(f"개념설명 데이터셋 저장됨: {concept_file}")
                
            except Exception as e:
                print(f"개념설명 데이터셋 생성 실패: {e}")
                if self.debug:
                    import traceback
                    print("❌ 상세 오류 정보:")
                    traceback.print_exc()
        
        print(f"=== 챕터 {chapter_num} 처리 완료 ===\n")
        
        # API 호출 간 대기시간
        time.sleep(3)

    def process_concepts_only(self):
        """모든 챕터의 개념설명 데이터셋만 처리"""
        print("개념설명 데이터셋 생성 시작...")
        
        # 답지 PDF를 한 번만 업로드
        self.upload_answer_sheet_once()
        
        # 1부터 20까지의 챕터 처리 (개념설명만)
        for chapter_num in range(1, 21):
            try:
                self.process_chapter_concepts_only(chapter_num)
            except Exception as e:
                print(f"챕터 {chapter_num} 개념설명 처리 중 오류 발생: {e}")
                continue
        
        print("모든 챕터 개념설명 처리 완료!")

    def process_chapter_concepts_only(self, chapter_num):
        """특정 챕터의 개념설명 데이터셋만 처리"""
        chapter_dir = self.base_path / f"Industrial_Tech_College_Prep_Workbook_chapter_{chapter_num}_pdf"
        
        if not chapter_dir.exists():
            print(f"챕터 {chapter_num} 디렉토리가 존재하지 않습니다: {chapter_dir}")
            return
        
        # 개념설명 파일이 이미 존재하는지 확인
        concept_file = self.response_path / f"chapter_{chapter_num}_concepts.json"
        if concept_file.exists():
            print(f"챕터 {chapter_num}의 개념설명 데이터셋이 이미 존재합니다. 건너뜁니다.")
            return
        
        print(f"\n=== 챕터 {chapter_num} 개념설명 처리 시작 ===")
        
        # 챕터 디렉토리 내의 PDF 파일들 찾기
        pdf_files = list(chapter_dir.glob("*.pdf"))
        
        if not pdf_files:
            print(f"챕터 {chapter_num}에서 PDF 파일을 찾을 수 없습니다.")
            return
        
        print(f"발견된 PDF 파일들: {[f.name for f in pdf_files]}")
        
        # PDF 파일들 업로드 (답지 제외)
        uploaded_files = []
        for pdf_file in pdf_files:
            try:
                # 답지가 아닌 파일만 업로드
                if not any(keyword in pdf_file.name.lower() for keyword in ["answer", "답", "해설", "solution"]):
                    uploaded_file = self.upload_pdf_file(pdf_file)
                    uploaded_files.append(uploaded_file)
            except Exception as e:
                print(f"파일 업로드 실패 {pdf_file.name}: {e}")
                continue
        
        # 답지 파일도 추가 (이미 업로드됨)
        if self.answer_sheet_file:
            uploaded_files.append(self.answer_sheet_file)
        
        if not uploaded_files:
            print(f"챕터 {chapter_num}에서 업로드된 파일이 없습니다.")
            return
        
        # 개념설명 데이터셋 생성
        print(f"\n--- 챕터 {chapter_num} 개념설명 데이터셋 생성 중 ---")
        try:
            concept_prompt = self.get_concept_explanation_prompt(chapter_num)
            
            if self.debug:
                print("🔍 개념설명 프롬프트 전송 중...")
                print("=" * 80)
                print("📝 개념설명 프롬프트 내용:")
                print("-" * 80)
                print(concept_prompt[:1000] + "..." if len(concept_prompt) > 1000 else concept_prompt)
                print("=" * 80)
                print("📤 Gemini API 호출 시작 (이 과정은 시간이 걸릴 수 있습니다)...")
            
            concept_response = self.model.generate_content(
                [concept_prompt] + uploaded_files,
                stream=True if self.debug else False,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=64000,
                    top_p=0.8,
                    top_k=40
                )
            )
            
            if self.debug:
                print("📥 Gemini 실시간 응답 스트리밍 시작...")
                print("=" * 80)
                print("🤖 실시간 Gemini 개념설명 응답:")
                print("-" * 80)
                
                full_response = ""
                for chunk in concept_response:
                    if chunk.text:
                        print(chunk.text, end="", flush=True)
                        full_response += chunk.text
                
                print("\n" + "=" * 80)
                print("📥 응답 스트리밍 완료")
                print("🧹 응답 정리 중...")
                print(f"📄 전체 응답 길이: {len(full_response)} 문자")
                
                # 스트리밍 응답을 단일 응답 객체로 변환
                class ResponseWrapper:
                    def __init__(self, text):
                        self.text = text
                
                concept_response = ResponseWrapper(full_response)
            else:
                print("📥 Gemini 응답 수신 완료")
                print("🧹 응답 정리 중...")
                print(f"📄 원본 응답 길이: {len(concept_response.text)} 문자")
            
            # JSON 응답에서 코드 블록 마커 제거
            cleaned_response = self.clean_json_response(concept_response.text)
            
            if self.debug:
                print(f"✂️ 정리된 응답 길이: {len(cleaned_response)} 문자")
                print("🔍 JSON 검증 중...")
            
            # 응답 검증
            try:
                concepts_data = json.loads(cleaned_response)
                concept_count = len(concepts_data)
                print(f"생성된 개념 질문-답변 쌍 수: {concept_count}개")
                
                if self.debug:
                    print(f"📊 개념 데이터 세부 정보:")
                    for i, concept in enumerate(concepts_data[:3]):  # 처음 3개만 표시
                        user_content = concept.get('messages', [{}])[1].get('content', 'N/A')[:50]
                        print(f"  개념 {i+1}: {user_content}...")
                    if concept_count > 3:
                        print(f"  ... 및 {concept_count-3}개 추가")
                
                if concept_count < 300:
                    print(f"경고: 개념 질문-답변 쌍이 300개 미만입니다 ({concept_count}개). 목표는 300개 이상입니다.")
                    
            except json.JSONDecodeError as e:
                print(f"경고: JSON 파싱 실패. 응답 형식을 확인해주세요. 오류: {e}")
                if self.debug:
                    print("💥 JSON 파싱 실패한 응답 내용 (처음 500자):")
                    print(cleaned_response[:500])
            
            # 응답 저장
            with open(concept_file, 'w', encoding='utf-8') as f:
                f.write(cleaned_response)
            
            print(f"개념설명 데이터셋 저장됨: {concept_file}")
            
        except Exception as e:
            print(f"개념설명 데이터셋 생성 실패: {e}")
            if self.debug:
                import traceback
                print("❌ 상세 오류 정보:")
                traceback.print_exc()
        
        print(f"=== 챕터 {chapter_num} 개념설명 처리 완료 ===\n")
        
        # API 호출 간 대기시간
        time.sleep(3)

    def process_all_chapters(self):
        """모든 챕터 처리"""
        print("데이터셋 생성 시작...")
        
        # 답지 PDF를 한 번만 업로드
        self.upload_answer_sheet_once()
        
        # 1부터 20까지의 챕터 처리
        for chapter_num in range(1, 21):
            try:
                self.process_chapter(chapter_num)
            except Exception as e:
                print(f"챕터 {chapter_num} 처리 중 오류 발생: {e}")
                continue
        
        print("모든 챕터 처리 완료!")

    def get_past_KICE_data(self):
        """KICE 기출문제 데이터셋 생성"""
        print("\n=== KICE 기출문제 데이터셋 생성 시작 ===")
        
        kice_base_path = Path("Industrial_Tech_KICE_June_Sept_Exams")
        
        if not kice_base_path.exists():
            print(f"KICE 디렉토리가 존재하지 않습니다: {kice_base_path}")
            return
        
        # KICE 디렉토리 내의 모든 하위 디렉토리 찾기
        kice_subdirs = [d for d in kice_base_path.iterdir() if d.is_dir()]
        
        if not kice_subdirs:
            print("KICE 하위 디렉토리를 찾을 수 없습니다.")
            return
        
        print(f"발견된 KICE 시험 디렉토리: {len(kice_subdirs)}개")
        for subdir in kice_subdirs:
            print(f"  - {subdir.name}")
        
        all_kice_data = []
        
        # 각 하위 디렉토리 처리
        for exam_dir in sorted(kice_subdirs):
            exam_name = exam_dir.name
            print(f"\n--- {exam_name} 처리 중 ---")
            
            # 기존 파일 존재 여부 확인
            output_file = self.response_path / f"KICE_{exam_name}.json"
            if output_file.exists():
                print(f"{exam_name} 데이터셋이 이미 존재합니다. 건너뜁니다.")
                continue
            
            # PDF 파일들 찾기
            pdf_files = list(exam_dir.glob("*.pdf"))
            
            if not pdf_files:
                print(f"{exam_name}에서 PDF 파일을 찾을 수 없습니다.")
                continue
            
            print(f"발견된 PDF 파일들: {[f.name for f in pdf_files]}")
            
            # PDF 파일들 업로드
            uploaded_files = []
            for pdf_file in pdf_files:
                try:
                    uploaded_file = self.upload_pdf_file(pdf_file)
                    uploaded_files.append(uploaded_file)
                except Exception as e:
                    print(f"파일 업로드 실패 {pdf_file.name}: {e}")
                    continue
            
            if not uploaded_files:
                print(f"{exam_name}에서 업로드된 파일이 없습니다.")
                continue
            
            # KICE 문제 추출
            try:
                kice_prompt = self.get_kice_exam_prompt(exam_name)
                
                if self.debug:
                    print("🔍 KICE 문제 추출 프롬프트 전송 중...")
                    print("=" * 80)
                    print("📝 KICE 프롬프트 내용:")
                    print("-" * 80)
                    print(kice_prompt[:1000] + "..." if len(kice_prompt) > 1000 else kice_prompt)
                    print("=" * 80)
                    print("📤 Gemini API 호출 시작...")
                
                kice_response = self.model.generate_content(
                    [kice_prompt] + uploaded_files,
                    stream=True if self.debug else False
                )
                
                if self.debug:
                    print("📥 Gemini 실시간 응답 스트리밍 시작...")
                    print("=" * 80)
                    print("🤖 실시간 Gemini KICE 응답:")
                    print("-" * 80)
                    
                    full_response = ""
                    for chunk in kice_response:
                        if chunk.text:
                            print(chunk.text, end="", flush=True)
                            full_response += chunk.text
                    
                    print("\n" + "=" * 80)
                    print("📥 응답 스트리밍 완료")
                    print("🧹 응답 정리 중...")
                    print(f"📄 전체 응답 길이: {len(full_response)} 문자")
                    
                    # 스트리밍 응답을 단일 응답 객체로 변환
                    class ResponseWrapper:
                        def __init__(self, text):
                            self.text = text
                    
                    kice_response = ResponseWrapper(full_response)
                else:
                    print("📥 Gemini 응답 수신 완료")
                    print("🧹 응답 정리 중...")
                    print(f"📄 원본 응답 길이: {len(kice_response.text)} 문자")
                
                # JSON 응답에서 코드 블록 마커 제거
                cleaned_response = self.clean_json_response(kice_response.text)
                
                if self.debug:
                    print(f"✂️ 정리된 응답 길이: {len(cleaned_response)} 문자")
                    print("🔍 JSON 검증 중...")
                
                # 응답 검증
                try:
                    kice_data = json.loads(cleaned_response)
                    problem_count = len(kice_data)
                    print(f"생성된 KICE 문제 수: {problem_count}개")
                    
                    if self.debug:
                        print(f"📊 KICE 문제별 세부 정보:")
                        for i, problem in enumerate(kice_data):
                            stimulus_count = len(problem.get('stimulus_box', {}))
                            exam_name_check = problem.get('EXAM_NAME', '?')
                            print(f"  문제 {i+1}: ID={problem.get('id', '?')}, 시험={exam_name_check[:30]}..., 보기수={stimulus_count}")
                    
                    # 전체 데이터에 추가
                    all_kice_data.extend(kice_data)
                    
                except json.JSONDecodeError as e:
                    print(f"경고: JSON 파싱 실패. 응답 형식을 확인해주세요. 오류: {e}")
                    if self.debug:
                        print("💥 JSON 파싱 실패한 응답 내용 (처음 500자):")
                        print(cleaned_response[:500])
                
                # 개별 시험 파일로 저장
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(cleaned_response)
                
                print(f"KICE {exam_name} 데이터셋 저장됨: {output_file}")
                
            except Exception as e:
                print(f"KICE {exam_name} 데이터셋 생성 실패: {e}")
                if self.debug:
                    import traceback
                    print("❌ 상세 오류 정보:")
                    traceback.print_exc()
            
            # API 호출 간 대기시간
            time.sleep(3)
        
        # 전체 KICE 데이터를 하나의 파일로도 저장
        if all_kice_data:
            all_kice_file = self.response_path / "KICE_all_exams.json"
            with open(all_kice_file, 'w', encoding='utf-8') as f:
                json.dump(all_kice_data, f, ensure_ascii=False, indent=2)
            
            print(f"\n전체 KICE 데이터셋 저장됨: {all_kice_file}")
            print(f"총 KICE 문제 수: {len(all_kice_data)}개")
        
        print("=== KICE 기출문제 데이터셋 생성 완료 ===")

    def get_csat_exam_prompt(self, exam_name):
        """CSAT 시험 데이터셋 생성을 위한 프롬프트"""
        return f"""
업로드된 PDF 파일들을 분석하여 다음 작업을 수행해주세요:

**목표**: {exam_name} 시험의 공업일반 과목 문제들을 JSON 형식으로 추출

**절대 중요**: PDF에서 찾을 수 있는 모든 공업일반 문제를 빠짐없이 추출해야 합니다. 답지는 PDF 후반부에 포함되어 있습니다.

**보기(stimulus_box) 처리 절대 필수사항**:
⚠️ 이 부분이 가장 중요합니다 ⚠️

1. 문제에서 "다음 중 옳은 것만을 <보기>에서 있는 대로 고른 것은?" 같은 표현이 있으면
2. 반드시 PDF에서 해당 문제의 <보기> 섹션을 찾아서
3. ㄱ, ㄴ, ㄷ, ㄹ, ㅁ 등의 각 보기 내용을 stimulus_box에 포함해야 합니다

**작업 지침**:
1. PDF에서 공업일반 과목의 모든 문제를 찾아주세요
2. 각 문제마다 <보기>가 있는지 꼼꼼히 확인하세요
3. <보기>가 있으면 반드시 ㄱ, ㄴ, ㄷ, ㄹ 내용을 추출하세요
4. 시험 정보를 정확히 파악하여 EXAM_NAME에 포함하세요
5. 찾은 모든 문제를 다음 JSON 형식으로 변환해주세요. **해설(explanation)은 제외합니다.**

```json
{{
  "id": "문제 번호",
  "EXAM_NAME": "시험제목",
  "question": "문제의 발문",
  "context": "문제의 제시문 (표, 글 등)",
  "stimulus_box": {{
    "ㄱ": "보기 ㄱ의 실제 내용을 여기에",
    "ㄴ": "보기 ㄴ의 실제 내용을 여기에"
  }},
  "options": {{
    "①": "선택지 1번 내용",
    "②": "선택지 2번 내용", 
    "③": "선택지 3번 내용",
    "④": "선택지 4번 내용",
    "⑤": "선택지 5번 내용"
  }},
  "answer": {{
    "correct_option": "정답 번호"
  }}
}}
```

**반드시 확인해야 할 사항들**:
✅ **explanation 필드는 절대로 포함하지 마세요.**
✅ 문제에 "~을 <보기>에서 고른 것은?" 표현이 있으면 → stimulus_box에 ㄱ,ㄴ,ㄷ,ㄹ 내용 필수 포함
✅ 보기가 진짜 없는 문제만 → stimulus_box를 빈 객체 {{}} 로 설정
✅ PDF에서 찾을 수 있는 모든 공업일반 문제 → 빠짐없이 포함
✅ 시험 정보를 정확히 파악하여 EXAM_NAME에 포함 (예: "2025학년도 대학수학능력시험")

**최종 점검**:
- 각 문제를 다시 한번 확인하여 <보기> 섹션이 있는데 stimulus_box가 비어있으면 안됩니다
- <보기>에서 고른다는 표현이 있는 문제는 100% stimulus_box에 ㄱ,ㄴ,ㄷ,ㄹ 내용이 있어야 합니다

응답은 반드시 유효한 JSON 배열 형식으로만 해주세요.
2025년부터 2020년까지의 모든 CSAT 문제를 포함해야 합니다.
"""

    def get_past_csat_data(self):
        """CSAT 기출문제 데이터셋 생성"""
        print("\n=== CSAT 기출문제 데이터셋 생성 시작 ===")
        
        csat_base_path = Path("2020_2025_past_csat_exam")
        
        if not csat_base_path.exists():
            print(f"CSAT 디렉토리가 존재하지 않습니다: {csat_base_path}")
            return
            
        pdf_files = list(csat_base_path.glob("*.pdf"))
        
        if not pdf_files:
            print("CSAT PDF 파일을 찾을 수 없습니다.")
            return
            
        print(f"발견된 CSAT PDF 파일들: {[f.name for f in pdf_files]}")
        
        all_csat_data = []
        
        for pdf_file in sorted(pdf_files, reverse=True):
            exam_name = pdf_file.stem
            print(f"\n--- {exam_name} 처리 중 ---")
            
            output_file = self.response_path / f"CSAT_{exam_name}.json"
            if output_file.exists():
                print(f"{exam_name} 데이터셋이 이미 존재합니다. 건너뜁니다.")
                continue

            try:
                uploaded_file = self.upload_pdf_file(pdf_file)
                
                csat_prompt = self.get_csat_exam_prompt(exam_name)
                
                if self.debug:
                    print("🔍 CSAT 문제 추출 프롬프트 전송 중...")
                    print("📤 Gemini API 호출 시작...")

                csat_response = self.model.generate_content(
                    [csat_prompt, uploaded_file],
                    stream=True if self.debug else False
                )
                
                full_response = ""
                if self.debug:
                    print("🤖 실시간 Gemini CSAT 응답:")
                    for chunk in csat_response:
                        if chunk.text:
                            print(chunk.text, end="", flush=True)
                            full_response += chunk.text
                    print("\n📥 응답 스트리밍 완료")
                else:
                    for chunk in csat_response:
                        full_response += chunk.text
                    print("📥 Gemini 응답 수신 완료")

                cleaned_response = self.clean_json_response(full_response)
                
                try:
                    csat_data = json.loads(cleaned_response)
                    print(f"생성된 CSAT 문제 수: {len(csat_data)}개")
                    all_csat_data.extend(csat_data)
                    
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(cleaned_response)
                    print(f"CSAT {exam_name} 데이터셋 저장됨: {output_file}")

                except json.JSONDecodeError as e:
                    print(f"경고: JSON 파싱 실패. 오류: {e}")

            except Exception as e:
                print(f"CSAT {exam_name} 데이터셋 생성 실패: {e}")

            time.sleep(3)

        if all_csat_data:
            all_csat_file = self.response_path / "CSAT_all_exams.json"
            with open(all_csat_file, 'w', encoding='utf-8') as f:
                json.dump(all_csat_data, f, ensure_ascii=False, indent=2)
            print(f"\n전체 CSAT 데이터셋 저장됨: {all_csat_file}")
            print(f"총 CSAT 문제 수: {len(all_csat_data)}개")
            
        print("=== CSAT 기출문제 데이터셋 생성 완료 ===")

def main():
    parser = argparse.ArgumentParser(description="Gemini를 사용하여 데이터셋 생성")
    parser.add_argument("--debug", action="store_true", help="디버그 모드 활성화")
    parser.add_argument("--model", choices=["pro", "flash"], default="pro", 
                       help="사용할 Gemini 모델 선택 (pro: gemini-2.5-pro, flash: gemini-2.0-flash-thinking-exp)")
    parser.add_argument("--kice", action="store_true", help="KICE 기출문제 데이터셋만 생성")
    parser.add_argument("--all", action="store_true", help="모든 데이터셋 생성 (챕터 + KICE)")
    parser.add_argument("--csat", action="store_true", help="CSAT 기출문제 데이터셋만 생성")
    parser.add_argument("--chapters", action="store_true", help="챕터별 개념설명 데이터셋만 생성")
    args = parser.parse_args()
    
    try:
        generator = DatasetGenerator(model_type=args.model, debug=args.debug)
        
        if args.debug:
            print("🐛 디버그 모드 활성화")
            print("📊 상세한 실행 정보가 표시됩니다")
        
        if args.csat:
            # CSAT 기출문제만 처리
            print("🎯 CSAT 기출문제 데이터셋 생성 모드")
            generator.get_past_csat_data()
        elif args.chapters:
            # 챕터별 개념설명만 처리
            print("🎯 챕터별 개념설명 데이터셋 생성 모드")
            generator.process_concepts_only()
        elif args.kice:
            # KICE 기출문제만 처리
            print("🎯 KICE 기출문제 데이터셋 생성 모드")
            generator.get_past_KICE_data()
        elif args.all:
            # 모든 데이터셋 처리
            print("🎯 전체 데이터셋 생성 모드 (챕터 + KICE + CSAT)")
            generator.process_all_chapters()
            generator.get_past_KICE_data()
            generator.get_past_csat_data()
        else:
            # 기본: 챕터만 처리
            print("🎯 챕터별 데이터셋 생성 모드")
            generator.process_all_chapters()
            
    except KeyError:
        print("GEMINI_API_KEY 환경변수가 설정되지 않았습니다.")
        print("다음 명령어로 API 키를 설정해주세요:")
        print("export GEMINI_API_KEY=your_api_key_here")
    except Exception as e:
        print(f"오류 발생: {e}")
        if args.debug:
            import traceback
            print("❌ 상세 오류 정보:")
            traceback.print_exc()

if __name__ == "__main__":
    main()
