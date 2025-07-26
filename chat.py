from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# 베이스 모델 로드
print("모델 로딩 중...")
base_model = AutoModelForCausalLM.from_pretrained(
    "LGAI-EXAONE/EXAONE-4.0-1.2B",
    torch_dtype=torch.float32,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("LGAI-EXAONE/EXAONE-4.0-1.2B")

# LoRA 모델 로드
print("LoRA 모델 로딩 중...")
model = PeftModel.from_pretrained(base_model, "./final_model")

def generate_with_reasoning(question, enable_reasoning=True):
    """추론 모드로 답변 생성"""
    messages = [
        {"role": "system", "content": "당신은 산업기술 분야의 전문가로서 객관식 문제를 분석하고 정확한 답변을 제공하는 교육 도우미입니다."},
        {"role": "user", "content": question}
    ]
    
    # EXAONE 4.0 추론 모드 활성화
    input_ids = tokenizer.apply_chat_template(
        messages, 
        tokenize=True, 
        add_generation_prompt=True, 
        return_tensors="pt",
        enable_thinking=enable_reasoning  # 추론 모드 활성화
    )
    
    # 추론을 위한 생성 설정
    generation_config = {
        "max_new_tokens": 1024,  # 추론을 위해 충분한 토큰
        "temperature": 0.6,      # EXAONE 권장 설정
        "top_p": 0.95,          # EXAONE 권장 설정
        "do_sample": True,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }
    
    print("답변 생성 중...")
    with torch.no_grad():
        output = model.generate(
            input_ids.to(model.device),
            **generation_config
        )
    
    # 생성된 텍스트 디코딩
    response = tokenizer.decode(output[0], skip_special_tokens=False)
    
    # 추론 부분과 최종 답변 분리
    if enable_reasoning and "<think>" in response and "</think>" in response:
        thinking_start = response.find("<think>")
        thinking_end = response.find("</think>") + 8
        thinking_part = response[thinking_start:thinking_end]
        final_answer = response[thinking_end:].strip()
        
        print("="*50)
        print("🧠 추론 과정:")
        print("="*50)
        print(thinking_part)
        print("\n" + "="*50)
        print("📝 최종 답변:")
        print("="*50)
        print(final_answer)
    else:
        print("="*50)
        print("📝 답변:")
        print("="*50)
        print(response)
    
    return response

def interactive_chat():
    """대화형 채팅"""
    print("="*60)
    print("🎓 EXAONE 4.0 산업기술 전문가 (추론 모드)")
    print("="*60)
    print("질문을 입력하세요 (종료: 'quit' 또는 'exit')")
    print("추론 모드 전환: 'reasoning on/off'")
    print("-"*60)
    
    enable_reasoning = True
    
    while True:
        user_input = input("\n❓ 질문: ").strip()
        
        if user_input.lower() in ['quit', 'exit', '종료']:
            print("👋 대화를 종료합니다.")
            break
        elif user_input.lower() == 'reasoning on':
            enable_reasoning = True
            print("🧠 추론 모드가 활성화되었습니다.")
            continue
        elif user_input.lower() == 'reasoning off':
            enable_reasoning = False
            print("💬 일반 모드로 전환되었습니다.")
            continue
        elif not user_input:
            continue
        
        try:
            generate_with_reasoning(user_input, enable_reasoning)
        except Exception as e:
            print(f"❌ 오류 발생: {e}")

# 테스트 실행
if __name__ == "__main__":
    print("🔧 테스트 질문으로 시작...")
    
    # 테스트 질문
    test_question = """다음 중 식품 공업의 특징으로 가장 적절한 것은?
    
① 대량 생산이 주를 이룬다
② 계절적 변동이 적다  
③ 원료의 부패성이 높다
④ 자동화 수준이 낮다
⑤ 표준화가 어렵다"""
    
    generate_with_reasoning(test_question, enable_reasoning=True)
    
    # 대화형 모드 시작
    print("\n" + "="*60)
    interactive_chat()