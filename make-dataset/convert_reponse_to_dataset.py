import json 
import os
import glob
from typing import List, Dict, Any, Optional


class concepts_data_type:
    """Concepts JSON 데이터를 파싱하는 클래스"""
    
    def __init__(self, json_data: List[Dict[str, Any]]):
        self.data = json_data
        
    def get_conversations(self) -> List[Dict[str, Any]]:
        """모든 대화 데이터를 반환"""
        return self.data
    
    def get_conversation_by_index(self, index: int) -> Optional[Dict[str, Any]]:
        """특정 인덱스의 대화 데이터를 반환"""
        if 0 <= index < len(self.data):
            return self.data[index]
        return None
    
    def get_messages_list(self) -> List[List[Dict[str, str]]]:
        """모든 대화의 메시지 리스트를 반환"""
        return [item["messages"] for item in self.data]
    
    def get_questions_and_answers(self) -> List[Dict[str, str]]:
        """질문과 답변만 추출하여 반환"""
        qa_pairs = []
        for conversation in self.data:
            messages = conversation["messages"]
            user_msg = next((msg["content"] for msg in messages if msg["role"] == "user"), "")
            assistant_msg = next((msg["content"] for msg in messages if msg["role"] == "assistant"), "")
            qa_pairs.append({
                "question": user_msg,
                "answer": assistant_msg
            })
        return qa_pairs


class problems_data_type:
    """Problems JSON 데이터를 파싱하는 클래스"""
    
    def __init__(self, json_data: List[Dict[str, Any]]):
        self.data = json_data
    
    def get_all_problems(self) -> List[Dict[str, Any]]:
        """모든 문제 데이터를 반환"""
        return self.data
    
    def get_problem_by_id(self, problem_id: str) -> Optional[Dict[str, Any]]:
        """특정 ID의 문제를 반환"""
        for problem in self.data:
            if problem.get("id") == problem_id:
                return problem
        return None
    
    def get_chapter_info(self, problem_index: int) -> Optional[Dict[str, str]]:
        """특정 문제의 챕터 정보를 반환"""
        if 0 <= problem_index < len(self.data):
            return self.data[problem_index].get("chapter_info")
        return None
    
    def get_question_text(self, problem_index: int) -> Optional[str]:
        """특정 문제의 질문 텍스트를 반환"""
        if 0 <= problem_index < len(self.data):
            return self.data[problem_index].get("question")
        return None
    
    def get_context(self, problem_index: int) -> Optional[str]:
        """특정 문제의 컨텍스트를 반환"""
        if 0 <= problem_index < len(self.data):
            return self.data[problem_index].get("context")
        return None
    
    def get_options(self, problem_index: int) -> Optional[Dict[str, str]]:
        """특정 문제의 선택지를 반환"""
        if 0 <= problem_index < len(self.data):
            return self.data[problem_index].get("options")
        return None
    
    def get_correct_answer(self, problem_index: int) -> Optional[Dict[str, str]]:
        """특정 문제의 정답과 해설을 반환"""
        if 0 <= problem_index < len(self.data):
            return self.data[problem_index].get("answer")
        return None
    
    def get_stimulus_box(self, problem_index: int) -> Optional[Dict[str, str]]:
        """특정 문제의 stimulus box 내용을 반환"""
        if 0 <= problem_index < len(self.data):
            return self.data[problem_index].get("stimulus_box")
        return None
    
    def extract_problem_data(self, problem_index: int) -> Optional[Dict[str, Any]]:
        """특정 문제의 모든 주요 데이터를 구조화하여 반환"""
        if 0 <= problem_index < len(self.data):
            problem = self.data[problem_index]
            return {
                "id": problem.get("id"),
                "chapter": problem.get("chapter_info", {}).get("chapter_number"),
                "chapter_title": problem.get("chapter_info", {}).get("chapter_title"),
                "question": problem.get("question"),
                "context": problem.get("context"),
                "options": problem.get("options"),
                "stimulus_box": problem.get("stimulus_box"),
                "correct_option": problem.get("answer", {}).get("correct_option"),
                "explanation": problem.get("answer", {}).get("explanation")
            }
        return None


class kice_data_type:
    """KICE JSON 데이터를 파싱하는 클래스"""
    
    def __init__(self, json_data: List[Dict[str, Any]]):
        self.data = json_data
    
    def get_all_problems(self) -> List[Dict[str, Any]]:
        """모든 KICE 문제 데이터를 반환"""
        return self.data
    
    def get_problem_by_id(self, problem_id: str) -> Optional[Dict[str, Any]]:
        """특정 ID의 KICE 문제를 반환"""
        for problem in self.data:
            if problem.get("id") == problem_id:
                return problem
        return None
    
    def get_exam_name(self, problem_index: int) -> Optional[str]:
        """특정 문제의 시험 정보를 반환"""
        if 0 <= problem_index < len(self.data):
            return self.data[problem_index].get("EXAM_NAME")
        return None
    
    def extract_kice_problem_data(self, problem_index: int) -> Optional[Dict[str, Any]]:
        """특정 KICE 문제의 모든 주요 데이터를 구조화하여 반환"""
        if 0 <= problem_index < len(self.data):
            problem = self.data[problem_index]
            return {
                "id": problem.get("id"),
                "exam_name": problem.get("EXAM_NAME"),
                "question": problem.get("question"),
                "context": problem.get("context"),
                "options": problem.get("options"),
                "stimulus_box": problem.get("stimulus_box"),
                "correct_option": problem.get("answer", {}).get("correct_option"),
                "explanation": problem.get("answer", {}).get("explanation")
            }
        return None


class DatasetGenerator:
    """JSON 데이터를 JSONL 형식의 파인튜닝 데이터셋으로 변환하는 클래스"""
    
    def __init__(self):
        self.dataset = []
    
    def convert_concepts_to_training_format(self, concepts_data: concepts_data_type) -> List[Dict[str, Any]]:
        """Concepts 데이터를 파인튜닝 형식으로 변환"""
        training_data = []
        
        for conversation in concepts_data.get_conversations():
            messages = conversation["messages"]
            
            # 기존 시스템, 사용자, 어시스턴트 메시지를 그대로 사용
            training_data.append({
                "messages": messages
            })
        
        return training_data
    
    def convert_problems_to_training_format(self, problems_data: problems_data_type) -> List[Dict[str, Any]]:
        """Problems 데이터를 파인튜닝 형식으로 변환"""
        training_data = []
        
        for i in range(len(problems_data.get_all_problems())):
            problem_info = problems_data.extract_problem_data(i)
            
            if not problem_info:
                continue
            
            # 문제 정보 추출
            chapter_title = problem_info.get("chapter_title", "")
            question = problem_info.get("question", "")
            context = problem_info.get("context", "")
            stimulus_box = problem_info.get("stimulus_box", {})
            options = problem_info.get("options", {})
            correct_option = problem_info.get("correct_option", "")
            explanation = problem_info.get("explanation", "")
            
            # stimulus_box 내용을 문자열로 변환
            stimulus_text = ""
            if stimulus_box:
                stimulus_items = []
                for key, value in stimulus_box.items():
                    stimulus_items.append(f"{key}. {value}")
                if stimulus_items:
                    stimulus_text = "<보기>\n" + "\n".join(stimulus_items)
            
            # 선택지를 문자열로 변환
            options_text = ""
            if options:
                options_items = []
                for key, value in options.items():
                    options_items.append(f"{key} {value}")
                options_text = "\n".join(options_items)
            
            # 사용자 메시지 구성
            user_content_parts = [f"{chapter_title} 단원 관련 문제입니다.", question]
            
            if context.strip():
                user_content_parts.append(context)
            
            if stimulus_text.strip():
                user_content_parts.append(stimulus_text)
            
            if options_text.strip():
                user_content_parts.append(options_text)
            
            user_content = "\n\n".join(user_content_parts)
            
            # 어시스턴트 응답 구성
            assistant_content = f"정답은 {correct_option}입니다.\n\n{explanation}"
            
            # 메시지 구조 생성
            messages = [
                {
                    "role": "system",
                    "content": "당신은 대한민국 고등학교에서 직업탐구 영역을 공부하고 있는 고등학생으로 문제를 보고 적절한 답을 고르시오."
                },
                {
                    "role": "user",
                    "content": user_content
                },
                {
                    "role": "assistant",
                    "content": assistant_content
                }
            ]
            
            training_data.append({
                "messages": messages
            })
        
        return training_data
    
    def convert_kice_to_training_format(self, kice_data: kice_data_type) -> List[Dict[str, Any]]:
        """KICE 데이터를 파인튜닝 형식으로 변환"""
        training_data = []
        
        for i in range(len(kice_data.get_all_problems())):
            problem_info = kice_data.extract_kice_problem_data(i)
            
            if not problem_info:
                continue
            
            # 문제 정보 추출
            exam_name = problem_info.get("exam_name", "")
            question = problem_info.get("question", "")
            context = problem_info.get("context", "")
            stimulus_box = problem_info.get("stimulus_box", {})
            options = problem_info.get("options", {})
            correct_option = problem_info.get("correct_option", "")
            explanation = problem_info.get("explanation", "")
            
            # stimulus_box 내용을 문자열로 변환
            stimulus_text = ""
            if stimulus_box:
                stimulus_items = []
                for key, value in stimulus_box.items():
                    stimulus_items.append(f"{key}. {value}")
                if stimulus_items:
                    stimulus_text = "<보기>\n" + "\n".join(stimulus_items)
            
            # 선택지를 문자열로 변환
            options_text = ""
            if options:
                options_items = []
                for key, value in options.items():
                    options_items.append(f"{key} {value}")
                options_text = "\n".join(options_items)
            
            # 사용자 메시지 구성 (KICE는 시험 정보와 문제 ID 포함)
            problem_id = problem_info.get("id", "")
            user_content_parts = [f"{exam_name} 모의고사 기출문제 {problem_id} 입니다.", question]
            
            if context.strip():
                user_content_parts.append(context)
            
            if stimulus_text.strip():
                user_content_parts.append(stimulus_text)
            
            if options_text.strip():
                user_content_parts.append(options_text)
            
            user_content = "\n\n".join(user_content_parts)
            
            # 어시스턴트 응답 구성
            assistant_content = f"정답은 {correct_option}입니다.\n\n{explanation}"
            
            # 메시지 구조 생성
            messages = [
                {
                    "role": "system",
                    "content": "당신은 대한민국 고등학교에서 직업탐구 영역을 공부하고 있는 고등학생으로 문제를 보고 적절한 답을 고르시오."
                },
                {
                    "role": "user",
                    "content": user_content
                },
                {
                    "role": "assistant",
                    "content": assistant_content
                }
            ]
            
            training_data.append({
                "messages": messages
            })
        
        return training_data
    
    def save_to_jsonl(self, data: List[Dict[str, Any]], output_file: str):
        """데이터를 JSONL 형식으로 저장"""
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')


def convert_per_chapter_concepts_to_dataset(input_file: str, generator: DatasetGenerator) -> List[Dict[str, Any]]:
    """챕터별 개념 설명 JSON을 파인튜닝 형식으로 변환"""
    with open(input_file, 'r', encoding='utf-8') as f:
        concepts_json = json.load(f)
    
    # concepts_json이 리스트인지 확인
    if not isinstance(concepts_json, list):
        print(f"  → 경고: {input_file}의 형식이 올바르지 않습니다. 리스트가 아닙니다.")
        return []
    
    training_data = []
    fixed_count = 0
    skipped_count = 0
    
    for i, concept in enumerate(concepts_json):
        try:
            # concept이 딕셔너리인지 확인
            if not isinstance(concept, dict):
                skipped_count += 1
                continue
            
            # Case 1: 올바른 messages 구조가 있는 경우
            if 'messages' in concept:
                messages = concept.get('messages')
                if isinstance(messages, list) and len(messages) >= 3:
                    # 메시지 구조 검증
                    valid_concept = True
                    for j, message in enumerate(messages):
                        if not isinstance(message, dict) or 'role' not in message or 'content' not in message:
                            valid_concept = False
                            break
                    
                    if valid_concept:
                        training_data.append(concept)
                        continue
                elif isinstance(messages, list) and len(messages) == 2:
                    # 2개 메시지인 경우 system 메시지 추가
                    system_message = {
                        "role": "system",
                        "content": "당신은 대한민국 고등학교에서 직업탐구 영역 공업일반 과목을 가르치는 교사입니다."
                    }
                    fixed_messages = [system_message] + messages
                    training_data.append({"messages": fixed_messages})
                    fixed_count += 1
                    continue
            
            # Case 2: messages 키가 없고 role, content가 직접 있는 경우
            elif 'role' in concept and 'content' in concept:
                # 이 경우 단일 메시지로 보이므로 완전한 대화 구조를 만들어야 함
                role = concept.get('role')
                content = concept.get('content')
                
                if role == 'user':
                    # user 메시지인 경우, 적절한 assistant 응답을 생성해야 하지만
                    # 여기서는 건너뛰는 것이 안전함
                    skipped_count += 1
                    continue
                elif role == 'assistant':
                    # assistant 메시지인 경우, 기본 user 질문을 생성
                    messages = [
                        {
                            "role": "system",
                            "content": "당신은 대한민국 고등학교에서 직업탐구 영역 공업일반 과목을 가르치는 교사입니다."
                        },
                        {
                            "role": "user", 
                            "content": "공업일반 과목에 대해 설명해주세요."
                        },
                        {
                            "role": "assistant",
                            "content": content
                        }
                    ]
                    training_data.append({"messages": messages})
                    fixed_count += 1
                    continue
            
            # Case 3: 다른 구조인 경우 건너뛰기
            skipped_count += 1
                
        except Exception as e:
            skipped_count += 1
            continue
    
    if fixed_count > 0:
        print(f"  → {fixed_count}개 항목 자동 수정됨")
    if skipped_count > 0:
        print(f"  → {skipped_count}개 항목 건너뜀")
    
    return training_data


def convert_per_chapter_problems_to_dataset(input_file: str, generator: DatasetGenerator) -> List[Dict[str, Any]]:
    """챕터별 problems 파일을 데이터셋으로 변환"""
    with open(input_file, 'r', encoding='utf-8') as f:
        problems_json = json.load(f)
    
    problems_parser = problems_data_type(problems_json)
    return generator.convert_problems_to_training_format(problems_parser)


def convert_kice_data_to_dataset(input_file: str, generator: DatasetGenerator) -> List[Dict[str, Any]]:
    """KICE 파일을 데이터셋으로 변환"""
    with open(input_file, 'r', encoding='utf-8') as f:
        kice_json = json.load(f)
    
    kice_parser = kice_data_type(kice_json)
    return generator.convert_kice_to_training_format(kice_parser)


def main():
    """모든 JSON 파일을 변환하여 하나의 JSONL 파일로 생성"""
    generator = DatasetGenerator()
    all_training_data = []
    
    # gemini_response 폴더의 모든 JSON 파일 찾기
    concepts_files = glob.glob('gemini_response/chapter_*_concepts.json')
    problems_files = glob.glob('gemini_response/chapter_*_problems.json')
    kice_files = glob.glob('gemini_response/KICE_*.json')
    
    print(f"발견된 concepts 파일: {len(concepts_files)}개")
    print(f"발견된 problems 파일: {len(problems_files)}개")
    print(f"발견된 KICE 파일: {len(kice_files)}개")
    
    # Concepts 파일들 처리
    for file_path in sorted(concepts_files):
        print(f"처리 중: {file_path}")
        try:
            concepts_data = convert_per_chapter_concepts_to_dataset(file_path, generator)
            all_training_data.extend(concepts_data)
            print(f"  → {len(concepts_data)}개 데이터 추가")
        except Exception as e:
            print(f"  → 오류 발생: {e}")
    
    # Problems 파일들 처리
    for file_path in sorted(problems_files):
        print(f"처리 중: {file_path}")
        try:
            problems_data = convert_per_chapter_problems_to_dataset(file_path, generator)
            all_training_data.extend(problems_data)
            print(f"  → {len(problems_data)}개 데이터 추가")
        except Exception as e:
            print(f"  → 오류 발생: {e}")
    
    # KICE 파일들 처리
    for file_path in sorted(kice_files):
        print(f"처리 중: {file_path}")
        try:
            kice_data = convert_kice_data_to_dataset(file_path, generator)
            all_training_data.extend(kice_data)
            print(f"  → {len(kice_data)}개 데이터 추가")
        except Exception as e:
            print(f"  → 오류 발생: {e}")
    
    # 최종 데이터셋 저장
    output_file = './dataset.jsonl'
    generator.save_to_jsonl(all_training_data, output_file)
    
    print(f"\n변환 완료!")
    print(f"총 {len(all_training_data)}개의 훈련 데이터가 '{output_file}'에 저장되었습니다.")
    
    # 첫 번째 데이터 예시 출력
    if all_training_data:
        print("\n=== 첫 번째 데이터 예시 ===")
        first_data = all_training_data[0]
        for message in first_data["messages"]:
            print(f"Role: {message['role']}")
            print(f"Content: {message['content'][:200]}...")
            print("-" * 50)


if __name__ == "__main__":
    main()
