#!/usr/bin/env python3
"""
Gemini 응답을 LLM 파인튜닝용 Chat 형식으로 변환하는 스크립트

사용법:
    python convert_data_selective.py [all|chapters_only|exams_only]
    
    - all: 모든 파일 처리 (기본값)
    - chapters_only: Chapter_{number}.json 파일들만 처리 (test 제외)
    - exams_only: CSAT_EXAM과 test 파일들만 처리
"""

import os
import json
import glob
from pathlib import Path
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GeminiToChatFormatConverter:
    def __init__(self, output_type="all"):
        """
        Gemini 응답을 Chat 형식으로 변환하는 클래스
        
        Args:
            output_type: "all", "chapters_only", "exams_only" 중 선택
        """
        self.base_dir = Path(__file__).parent
        self.input_dir = self.base_dir / "gemini_responses_from_flash"
        self.output_type = output_type
        
        # 출력 파일명 설정
        if output_type == "chapters_only":
            self.output_file = self.base_dir / "chapters_only_from_gemini_flash.jsonl"
        elif output_type == "exams_only":
            self.output_file = self.base_dir / "exams_only_from_gemini_flash.jsonl"
        else:
            self.output_file = self.base_dir / "combine_from_gemini_flash.jsonl"
        
    def load_json_file(self, file_path):
        """JSON 파일 로드"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"✅ 로드 완료: {file_path.name} ({len(data) if isinstance(data, list) else 1}개 항목)")
            return data
        except Exception as e:
            logger.error(f"❌ 파일 로드 실패 {file_path.name}: {str(e)}")
            return None
    
    def convert_problem_to_chat(self, problem_data):
        """문제 형식 데이터를 Chat 형식으로 변환"""
        chat_messages = []
        
        # 1. 문제 해결 대화 생성
        problem_text = self.format_problem_text(problem_data)
        
        chat_data = {
            "messages": [
                {
                    "role": "system",
                    "content": "당신은 산업기술 분야의 전문가로서 객관식 문제를 분석하고 정확한 답변을 제공하는 교육 도우미입니다."
                },
                {
                    "role": "user", 
                    "content": problem_text
                },
                {
                    "role": "assistant",
                    "content": self.format_answer_text(problem_data)
                }
            ]
        }
        chat_messages.append(chat_data)
        
        # 2. 개념 설명 대화 생성 (context가 있는 경우)
        context = problem_data.get("context", "")
        if context and isinstance(context, str) and len(context.strip()) > 50:
            concept_chat = self.create_concept_explanation_chat(problem_data)
            if concept_chat:
                chat_messages.append(concept_chat)
        
        return chat_messages
    
    def format_problem_text(self, problem_data):
        """문제 텍스트 포맷팅"""
        problem_text = ""
        
        # 챕터 정보 추가
        if problem_data.get("chapter_info"):
            chapter_info = problem_data["chapter_info"]
            problem_text += f"**{chapter_info.get('chapter_number', '')} - {chapter_info.get('chapter_title', '')}**\n\n"
        
        # 제시문 추가
        if problem_data.get("context"):
            problem_text += f"**제시문:**\n{problem_data['context']}\n\n"
        
        # 문제 발문 추가
        if problem_data.get("question"):
            problem_text += f"**문제:** {problem_data['question']}\n\n"
        
        # 보기 추가 (stimulus_box)
        if problem_data.get("stimulus_box"):
            problem_text += "**<보기>**\n"
            for key, value in problem_data["stimulus_box"].items():
                problem_text += f"{key}. {value}\n"
            problem_text += "\n"
        
        # 선택지 추가
        if problem_data.get("options"):
            problem_text += "**선택지:**\n"
            for key, value in problem_data["options"].items():
                problem_text += f"{key} {value}\n"
        
        return problem_text.strip()
    
    def format_answer_text(self, problem_data):
        """답변 텍스트 포맷팅"""
        answer_info = problem_data.get("answer", {})
        
        if not answer_info.get("answer_available", False):
            return "죄송합니다. 이 문제에 대한 정답 정보가 제공되지 않았습니다."
        
        correct_option = answer_info.get("correct_option", "unknown")
        explanation = answer_info.get("explanation", "")
        
        answer_text = f"정답: **{correct_option}**\n\n"
        
        if explanation and explanation != "정답이 제공되지 않음":
            answer_text += f"**해설:**\n{explanation}"
        else:
            answer_text += "해설: 추가적인 해설이 필요합니다."
        
        return answer_text
    
    def create_concept_explanation_chat(self, problem_data):
        """문제의 개념을 설명하는 추가 채팅 생성"""
        context = problem_data.get("context", "")
        chapter_info = problem_data.get("chapter_info", {})
        
        # context가 문자열이 아닌 경우 문자열로 변환
        if not isinstance(context, str):
            context = str(context) if context else ""
        
        if len(context.strip()) < 50:
            return None
        
        # 제시문에서 주요 개념 추출하여 질문 생성
        question_content = f"{chapter_info.get('chapter_title', '산업기술')} 분야에서 다음 내용을 설명해주세요:\n\n{context}"
        
        return {
            "messages": [
                {
                    "role": "system", 
                    "content": "당신은 산업기술 전문가입니다."
                },
                {
                    "role": "user",
                    "content": question_content
                },
                {
                    "role": "assistant",
                    "content": f"이는 {chapter_info.get('chapter_title', '산업기술')} 분야의 중요한 내용입니다. {context}\n\n이러한 개념들은 실제 산업 현장에서 중요한 역할을 하며, 관련 문제를 해결하는 데 필수적인 지식입니다."
                }
            ]
        }
    
    def process_file(self, file_path):
        """개별 파일 처리"""
        data = self.load_json_file(file_path)
        if not data:
            return []
        
        chat_data_list = []
        
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    # 이미 chat 형식인 경우 (messages 키가 있는 경우)
                    if "messages" in item:
                        # 순수한 messages 형식으로 정리 (다른 메타데이터 제거)
                        clean_chat = {
                            "messages": item["messages"]
                        }
                        chat_data_list.append(clean_chat)
                    else:
                        # 문제 형식인 경우 변환
                        converted_chats = self.convert_problem_to_chat(item)
                        chat_data_list.extend(converted_chats)
        
        logger.info(f"  📊 변환 결과: {len(chat_data_list)}개 chat 데이터 생성")
        return chat_data_list
    
    def get_filtered_files(self):
        """출력 타입에 따라 파일 필터링"""
        json_files = list(self.input_dir.glob("*.json"))
        
        if self.output_type == "chapters_only":
            # Chapter_{number}.json 파일들만 (test 제외)
            filtered_files = [
                f for f in json_files 
                if f.name.startswith("Chapter_") and not f.name.endswith("_test.json")
            ]
            # 챕터 번호 순으로 정렬
            def extract_chapter_number(filename):
                import re
                match = re.search(r'Chapter_(\d+)\.json$', filename)
                if match:
                    return int(match.group(1))
                return 999
            
            filtered_files.sort(key=lambda x: extract_chapter_number(x.name))
            
        elif self.output_type == "exams_only":
            # CSAT_EXAM과 test 파일들만
            filtered_files = [
                f for f in json_files 
                if "CSAT_EXAM" in f.name or f.name.endswith("_test.json")
            ]
            # 날짜와 타입 순으로 정렬
            import re
            def sort_key(file_path):
                filename = file_path.name
                if "CSAT_EXAM" in filename:
                    match = re.search(r'(\d{2})_(\d{2})_CSAT_EXAM', filename)
                    if match:
                        year, month = int(match.group(1)), int(match.group(2))
                        return (0, year, month, filename)  # CSAT가 먼저
                elif filename.endswith("_test.json"):
                    match = re.search(r'Chapter_(\d+)_test', filename)
                    if match:
                        chapter_num = int(match.group(1))
                        return (1, chapter_num, 0, filename)  # test가 나중에
                return (2, 999, 0, filename)
            
            filtered_files.sort(key=sort_key)
            
        else:  # "all"
            # 모든 파일 (기존 로직)
            chapter_files = []
            csat_files = []
            
            for file_path in json_files:
                if file_path.name.startswith("Chapter_"):
                    chapter_files.append(file_path)
                elif "CSAT_EXAM" in file_path.name:
                    csat_files.append(file_path)
            
            # Chapter 파일들 정렬 (숫자 순)
            def extract_chapter_number(filename):
                import re
                match = re.search(r'Chapter_(\d+)', filename)
                if match:
                    return int(match.group(1))
                return 999
            
            chapter_files.sort(key=lambda x: (extract_chapter_number(x.name), x.name))
            
            # CSAT 파일들 정렬 (연도_월 순)
            def extract_csat_date(filename):
                import re
                match = re.search(r'(\d{2})_(\d{2})_CSAT_EXAM', filename)
                if match:
                    year, month = int(match.group(1)), int(match.group(2))
                    return (year, month)
                return (99, 99)
            
            csat_files.sort(key=lambda x: extract_csat_date(x.name))
            
            filtered_files = chapter_files + csat_files
        
        return filtered_files
    
    def convert_all_files(self):
        """모든 파일을 처리하여 JSONL 형식으로 저장"""
        filtered_files = self.get_filtered_files()
        all_chat_data = []
        
        # 타입별 로그 메시지
        type_messages = {
            "all": "🚀 모든 Gemini 응답을 Chat 형식으로 변환 시작",
            "chapters_only": "📚 Chapter 파일들만 Chat 형식으로 변환 시작 (test 제외)",
            "exams_only": "📝 시험 파일들만 Chat 형식으로 변환 시작 (CSAT + test)"
        }
        
        logger.info(f"\n{'='*70}")
        logger.info(type_messages.get(self.output_type, "🚀 변환 시작"))
        logger.info(f"{'='*70}")
        
        logger.info(f"📋 처리할 파일 목록 ({len(filtered_files)}개):")
        for i, file_path in enumerate(filtered_files, 1):
            logger.info(f"  {i:2d}. {file_path.name}")
        
        total_files = len(filtered_files)
        
        for i, file_path in enumerate(filtered_files, 1):
            logger.info(f"\n📂 [{i:2d}/{total_files}] {file_path.name} 처리 중...")
            logger.info(f"{'─'*50}")
            
            chat_data_list = self.process_file(file_path)
            all_chat_data.extend(chat_data_list)
            
            logger.info(f"✅ {file_path.name} 완료 (누적: {len(all_chat_data)}개 채팅)")
        
        # JSONL 파일로 저장
        self.save_as_jsonl(all_chat_data)
        
        # 통계 출력
        self.print_statistics(all_chat_data)
        
        return all_chat_data
    
    def save_as_jsonl(self, chat_data_list):
        """Chat 데이터를 JSONL 형식으로 저장"""
        try:
            with open(self.output_file, 'w', encoding='utf-8') as f:
                for chat_data in chat_data_list:
                    json_line = json.dumps(chat_data, ensure_ascii=False)
                    f.write(json_line + '\n')
            
            logger.info(f"\n💾 저장 완료: {self.output_file}")
            logger.info(f"📊 총 {len(chat_data_list):,}개의 채팅 데이터가 저장되었습니다.")
            
        except Exception as e:
            logger.error(f"❌ 파일 저장 실패: {str(e)}")
    
    def print_statistics(self, chat_data_list):
        """변환 결과 통계 출력"""
        total_chats = len(chat_data_list)
        
        # 시스템 메시지별 분류
        system_messages = {}
        for chat in chat_data_list:
            messages = chat.get("messages", [])
            if messages and messages[0].get("role") == "system":
                system_content = messages[0].get("content", "")
                if "전문가" in system_content:
                    key = "산업기술 전문가"
                elif "교육 도우미" in system_content:
                    key = "교육 도우미 (문제 해결)"
                else:
                    key = "기타"
                system_messages[key] = system_messages.get(key, 0) + 1
        
        logger.info(f"\n{'='*70}")
        logger.info(f"📊 변환 결과 통계 ({self.output_type})")
        logger.info(f"{'='*70}")
        logger.info(f"🔢 총 채팅 데이터 수: {total_chats:,}개")
        logger.info(f"\n📋 채팅 유형별 분포:")
        for msg_type, count in system_messages.items():
            percentage = (count / total_chats) * 100
            logger.info(f"  • {msg_type}: {count:,}개 ({percentage:.1f}%)")
        
        # 파일 크기 정보
        if self.output_file.exists():
            file_size = self.output_file.stat().st_size
            file_size_mb = file_size / (1024 * 1024)
            logger.info(f"\n💾 출력 파일 정보:")
            logger.info(f"  • 파일명: {self.output_file.name}")
            logger.info(f"  • 파일 크기: {file_size_mb:.2f} MB")
            logger.info(f"  • 평균 라인당 크기: {file_size / total_chats:.0f} bytes")

def main():
    """메인 실행 함수"""
    import sys
    
    # 사용법 출력
    def print_usage():
        print("=" * 70)
        print("🤖 Gemini 응답을 LLM 파인튜닝용 Chat 형식으로 변환")
        print("=" * 70)
        print("\n📖 사용법:")
        print("  python convert_gemini_to_chat.py [옵션]")
        print("\n🔧 옵션:")
        print("  all           - 모든 파일 처리 (기본값)")
        print("  chapters_only - Chapter_{number}.json 파일들만 처리 (test 제외)")  
        print("  exams_only    - CSAT_EXAM과 test 파일들만 처리")
        print("\n📁 출력 파일:")
        print("  all           → combine_from_gemini_flash.jsonl")
        print("  chapters_only → chapters_only_from_gemini_flash.jsonl")
        print("  exams_only    → exams_only_from_gemini_flash.jsonl")
        print("\n💡 예시:")
        print("  python convert_gemini_to_chat.py")
        print("  python convert_gemini_to_chat.py chapters_only")
        print("  python convert_gemini_to_chat.py exams_only")
        print()
    
    # 명령행 인수 처리
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help', 'help']:
        print_usage()
        return
    
    output_type = "all"
    if len(sys.argv) > 1:
        output_type = sys.argv[1]
    
    if output_type not in ["all", "chapters_only", "exams_only"]:
        print("❌ 잘못된 옵션입니다.")
        print_usage()
        return
    
    converter = GeminiToChatFormatConverter(output_type=output_type)
    
    try:
        # 입력 디렉토리 확인
        if not converter.input_dir.exists():
            logger.error(f"❌ 입력 디렉토리가 존재하지 않습니다: {converter.input_dir}")
            return
        
        # 변환 실행
        chat_data = converter.convert_all_files()
        
        logger.info(f"\n🎉 변환 작업이 성공적으로 완료되었습니다!")
        logger.info(f"💾 결과 파일: {converter.output_file}")
        logger.info(f"📊 총 데이터 수: {len(chat_data):,}개")
        
    except KeyboardInterrupt:
        logger.info(f"\n⚠️ 사용자에 의해 작업이 중단되었습니다.")
    except Exception as e:
        logger.error(f"\n❌ 예상치 못한 오류가 발생했습니다: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
