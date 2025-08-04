from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import json
import re
import argparse
from tqdm import tqdm
import torch

# ------- Default Model Settings for Transformers ------- # 
model_name = "./fused_model" 
#model_name = "LGAI-EXAONE/EXAONE-4.0-1.2B"

# Load fused model (already contains adapter weights)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,  # Set to bf16
    device_map="auto",
    low_cpu_mem_usage=True
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

def load_dataset(file_path):
    """JSON 파일에서 데이터셋을 로드하고 연도별로 그룹화 및 파싱합니다."""
    with open(file_path, 'r', encoding='utf-8') as f:
        problems = json.load(f)
    
    yearly_data = {}
    option_map = {'①': 1, '②': 2, '③': 3, '④': 4, '⑤': 5}

    for problem in problems:
        try:
            # EXAM_NAME에서 연도 정보 추출
            match = re.search(r'(\d{4})', problem.get('EXAM_NAME', ''))
            if not match:
                continue
            
            year = int(match.group(1))
            if year not in yearly_data:
                yearly_data[year] = {'year': year, 'problems': []}
            
            # 점수 파싱
            score = 3 if '[3점]' in problem.get('question', '') else 2

            # 컨텍스트와 보기(stimulus_box)를 질문에 포함
            full_question_text = ""
            if problem.get('context'):
                full_question_text += problem['context'] + "\n\n"
            
            if problem.get('stimulus_box') and isinstance(problem['stimulus_box'], dict):
                stimulus_text = "\n".join([f"{key}. {value}" for key, value in problem['stimulus_box'].items()])
                if stimulus_text:
                    full_question_text += "<보기>\n" + stimulus_text + "\n\n"
            
            full_question_text += problem.get('question', '')

            # 선택지 파싱
            options_list = []
            options_dict = problem.get('options', {})
            if isinstance(options_dict, dict):
                # 키 순서 보장을 위해 option_map 순서대로 정렬
                sorted_keys = sorted(options_dict.keys(), key=lambda k: list(option_map.keys()).index(k) if k in option_map else 99)
                options_list = [options_dict[key] for key in sorted_keys]

            # 정답 파싱
            answer_key = problem.get('answer', {}).get('correct_option')
            answer = option_map.get(answer_key)

            if answer is None:
                continue

            yearly_data[year]['problems'].append({
                'id': problem.get('id', 'N/A'),
                'question': full_question_text,
                'options': options_list,
                'answer': answer,
                'score': score
            })

        except Exception as e:
            print(f"경고: 문제 파싱 중 오류 발생, 건너뜁니다. 오류: {e}, 문제 ID: {problem.get('id')}")
            continue
    
    return sorted(yearly_data.values(), key=lambda x: x['year'])

def get_model_answer(prompt):
    system_prompt = """당신은 대한민국 수능 직업탐구 영역 '공업 일반' 과목에 정통한 전문가입니다.
주어진 문제를 신중히 분석하고, 아래 JSON 형식으로 답변하세요.

{
    "정답": 정답번호,
    "해설": 문제해설
}

정답은 반드시 1, 2, 3, 4, 5 중 하나의 숫자여야 합니다."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    
    # Enable reasoning mode with enable_thinking=True
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        presence_penalty=1.5,
        enable_thinking=True
    )

    # Create streamer to show reasoning process
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    print("\n🤔 모델 사고 과정:")
    print("-" * 60)
    
    # Generate with reasoning parameters and streaming
    output = model.generate(
        input_ids.to(model.device),
        max_new_tokens=12000,
        do_sample=True,
        temperature=0.65,
        top_p=0.95,
        streamer=streamer
    )
    
    print("-" * 60)
    
    response = tokenizer.decode(output[0], skip_special_tokens=False)
    
    # Remove the input from the response to get only the generated part
    prompt_text = tokenizer.decode(input_ids[0], skip_special_tokens=False)
    if response.startswith(prompt_text):
        response = response[len(prompt_text):].strip()
    
    return response

def extract_answer(text):
    """모델의 JSON 응답에서 정답 번호를 추출합니다."""
    try:
        # Extract content after </think> tag (the final answer)
        think_pattern = r'</think>(.*?)$'
        think_match = re.search(think_pattern, text, re.DOTALL)
        if think_match:
            final_response = think_match.group(1).strip()
        else:
            final_response = text
        
        # 1. 가장 마지막에 나타나는 JSON 블록을 파싱 (가장 가능성 높은 최종 답변)
        try:
            json_matches = re.findall(r'\{[^{}]*\}', final_response, re.DOTALL)
            if json_matches:
                # 가장 마지막 JSON 블록을 사용
                json_str = json_matches[-1]
                data = json.loads(json_str)
                if "정답" in data and str(data["정답"]).isdigit() and 1 <= int(data["정답"]) <= 5:
                    return int(data["정답"])
        except (json.JSONDecodeError, ValueError):
            pass

        # 2. '정답' 키워드와 함께 명시된 숫자 추출 (가장 마지막 매칭)
        # 예: "정답": 3, 정답: 3, 정답은 3
        answer_patterns = [
            r'"정답"\s*:\s*"?([1-5])"?',
            r'정답은?\s*[:\s]*([1-5])'
        ]
        
        best_match = None
        for pattern in answer_patterns:
            matches = list(re.finditer(pattern, final_response))
            if matches:
                best_match = matches[-1] # 가장 마지막 매칭 사용
        
        if best_match:
            return int(best_match.group(1))

        # 3. 문맥에서 벗어난 단일 숫자 추출
        # 모델이 숫자만 덩그러니 반환하는 경우
        final_numbers = re.findall(r'^\s*([1-5])\s*$', final_response, re.MULTILINE)
        if final_numbers:
            return int(final_numbers[-1])

    except (ValueError, TypeError):
        pass
    
    return None

def solve_and_grade(yearly_data, debug=False):
    """연도별 문제를 풀고 채점합니다."""
    total_score = 0
    correct_count = 0
    total_questions = len(yearly_data['problems'])
    
    # 로그 파일 초기화
    with open('./log.txt', 'w', encoding='utf-8') as log_file:
        log_file.write(f"===== {yearly_data['year']}년도 오답 및 오류 로그 =====\n\n")
    
    print(f"\n===== {yearly_data['year']}년도 문제 풀이 시작 =====")
    
    # 디버그 모드에서는 tqdm을 비활성화합니다.
    if debug:
        problems_iter = yearly_data['problems']
    else:
        problems_iter = tqdm(yearly_data['problems'], desc=f"{yearly_data['year']}년도 진행률")
    
    for i, problem in enumerate(problems_iter, 1):
        question_text = problem['question']
        if problem.get('options'):
            options_text = "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(problem['options'])])
            question_text += "\n" + options_text
        
        prompt = f"다음 문제를 신중히 분석하고 JSON 형식으로 답변해주세요.\n\n{question_text}"
        
        model_response = get_model_answer(prompt)
        model_answer = extract_answer(model_response)
        correct_answer = problem['answer']

        is_correct = model_answer == correct_answer
        if is_correct:
            total_score += problem['score']
            correct_count += 1

        # 실시간 채점 결과 출력 (디버그 모드가 아닐 때도)
        result_symbol = "✓" if is_correct else "✗"
        print(f"문제 {i:2d}/{total_questions} [{problem['id']}]: {result_symbol} (모델답: {model_answer}, 정답: {correct_answer}) | 현재 점수: {total_score}/50점 (정답률: {correct_count}/{i}, {correct_count/i*100:.1f}%)")

        # 오답이거나 모델답이 None인 경우 로그에 기록
        if not is_correct or model_answer is None:
            with open('./log.txt', 'a', encoding='utf-8') as log_file:
                log_file.write(f"--- 문제 {i} [{problem['id']}] ---\n")
                log_file.write(f"상태: {'파싱 실패' if model_answer is None else '오답'}\n")
                log_file.write(f"모델 답: {model_answer}\n")
                log_file.write(f"정답: {correct_answer}\n")
                log_file.write(f"문제:\n{question_text}\n\n")
                log_file.write(f"모델 응답:\n{model_response}\n")
                log_file.write("="*80 + "\n\n")

        if debug:
            print(f"\n--- 문제 {problem['id']} 상세 ---")
            print(f"문제:\n{question_text}")
            print(f"모델 응답: {model_response}")
            print(f"모델 추출 답: {model_answer}, 정답: {correct_answer}")
            print(f"채점 결과: {'정답' if is_correct else '오답'} (점수: {problem['score'] if is_correct else 0})")
            input("계속하려면 Enter를 누르세요...")

    print(f"\n===== {yearly_data['year']}년도 최종 채점 결과 =====")
    print(f"총 {total_questions}문제 중 {correct_count}문제 정답 ({correct_count/total_questions*100:.1f}%)")
    print(f"총점: {total_score}점 / 50점 만점 ({total_score/50*100:.1f}%)")
    print(f"오답 및 오류 로그가 './log.txt'에 저장되었습니다.")
    return total_score

def main():
    parser = argparse.ArgumentParser(description="CSAT 벤치마크 스크립트")
    parser.add_argument('--debug', action='store_true', help='디버그 모드를 활성화하여 문제별 진행 상황과 모델 응답을 확인합니다.')
    args = parser.parse_args()

    dataset = load_dataset('benchmark/full_csat_dataset.json')
    
    # 2025년 데이터만 필터링하여 실행
    found_2025 = False
    for year_data in dataset:
        # 'year' 키의 값이 문자열일 수 있으므로 int로 변환하여 비교합니다.
        if int(year_data.get('year', 0)) == 2025:
            solve_and_grade(year_data, debug=args.debug)
            found_2025 = True
            break
    
    if not found_2025:
        print("2025년도 데이터를 찾을 수 없습니다.")

if __name__ == "__main__":
    main()