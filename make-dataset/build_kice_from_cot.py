#!/usr/bin/env python3
# build_kice_to_cot.py
# ─────────────────────────────────────────────────────────────
# 사용 예:
#   export GEMINI_API_KEY="AIza…"
#   uv run build_kice_to_cot.py \
#         --src KICE_all_exams.json \
#         --out kice_cot.jsonl \
#         --gemini-out gemini_reply.jsonl      # 선택: Gemini 호출 저장
#
# --gemini-out 를 주면 ► 각 문항에 대해 Gemini-Pro 응답(검증/추가 설명 등)을
#                        따로 jsonl 로 저장합니다.
# ─────────────────────────────────────────────────────────────
import argparse
import json
import re
import time
from pathlib import Path
from typing import List, Dict

# ↓↓↓ ①  공통 상수
SYSTEM_PROMPT = (
    "당신은 대한민국 수능 직업탐구 영역 '공업 일반' 과목에 정통한 전문가입니다."
)
CIRCLES = "①②③④⑤"

# ─────────────────────────────────────────────────────────────
# ↓↓↓ ②  유틸
# ─────────────────────────────────────────────────────────────
def kor_choice(n: str) -> str:
    return n if re.fullmatch(r"[①-⑤]", n) else CIRCLES[int(n) - 1]

def guard_strip(txt: str | None) -> str:
    return txt.strip() if txt else ""

# ─────────────────────────────────────────────────────────────
# ↓↓↓ ③  user 메시지 빌드 (세분화)
# ─────────────────────────────────────────────────────────────
def build_header(row: Dict) -> str:
    return f"{row['EXAM_NAME']} {row['id']}"

def build_question(row: Dict) -> str:
    return guard_strip(row["question"])

def build_context(row: Dict) -> str:
    return guard_strip(row.get("context"))

def build_stimulus_box(row: Dict) -> str:
    box = row.get("stimulus_box") or {}
    if not box:
        return ""
    lines = ["<보기>"]
    for k, v in box.items():
        lines.append(f"{k}  {v}")
    return "\n".join(lines)

def build_options(row: Dict) -> str:
    opts = row.get("options") or {}
    return "\n".join(f"{k}  {v}" for k, v in opts.items())

def build_user_prompt(row: Dict) -> str:
    parts = [
        build_header(row),
        build_question(row),
        build_context(row),
        build_stimulus_box(row),
        build_options(row),
    ]
    return "\n\n".join(p for p in parts if p).strip()

# ─────────────────────────────────────────────────────────────
# ↓↓↓ ④  assistant 메시지 빌드
# ─────────────────────────────────────────────────────────────
def extract_answer(row: Dict) -> str:
    return kor_choice(guard_strip(row["answer"]["correct_option"]))

def extract_explanation(row: Dict) -> str:
    return guard_strip(row["answer"]["explanation"])

def build_assistant_message(row: Dict) -> str:
    return f"정답: {extract_answer(row)}\n\n해설: {extract_explanation(row)}"

# ─────────────────────────────────────────────────────────────
# ↓↓↓ ⑤  row → ChatGPT(CoT) 포맷
# ─────────────────────────────────────────────────────────────
def convert_row(row: Dict) -> Dict:
    return {
        "id": f"{row['EXAM_NAME']}_{row['id']}",
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": build_user_prompt(row)},
            {"role": "assistant", "content": build_assistant_message(row)},
        ],
    }

# ─────────────────────────────────────────────────────────────
# ↓↓↓ ⑥  Gemini 호출 파트
# ─────────────────────────────────────────────────────────────
try:
    import google.generativeai as genai
except ModuleNotFoundError:
    genai = None  # 라이브러리 미설치 시에도 변환 기능은 동작하게 함.

def init_gemini(api_key: str, model_name: str = "models/gemini-2.5-flash-lite"):
    """Gemini-Pro 초기화 & 모델 핸들 반환."""
    if not genai:
        raise RuntimeError("google-generativeai 라이브러리가 설치돼 있지 않습니다.")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)

def _convert_messages_for_gemini(msgs: List[Dict]) -> List[Dict]:
    """
    Gemini API 는 role = {user, model} 만 허용.
    -> system 프롬프트를 user 0 번째 메시지 앞에 합침.
    """
    new_msgs: List[Dict] = []
    system_text = ""
    for m in msgs:
        if m["role"] == "system":
            system_text += m["content"] + "\n\n"
        else:
            role = "user" if m["role"] == "user" else "model"
            text = (system_text + m["content"]).strip() if system_text else m["content"]
            new_msgs.append({"role": role, "parts": text})
            system_text = ""  # 한 번 합치면 초기화
    return new_msgs

def gemini_generate(model, msgs: List[Dict], temp: float = 0.3) -> str:
    """Gemini-Pro 대화 생성 -> 텍스트 str 반환."""
    try:
        g_msgs = _convert_messages_for_gemini(msgs)
        resp = model.generate_content(g_msgs, generation_config={"temperature": temp})
        return resp.text.strip()
    except Exception as e:
        return f"[Gemini Error] {e}"

# ─────────────────────────────────────────────────────────────
# ↓↓↓ ⑦  I/O
# ─────────────────────────────────────────────────────────────
def read_source(path: Path) -> List[Dict]:
    with path.open(encoding="utf-8") as rf:
        return json.load(rf)

def dump_jsonl(items: List[Dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as wf:
        for obj in items:
            wf.write(json.dumps(obj, ensure_ascii=False) + "\n")

# ─────────────────────────────────────────────────────────────
# ↓↓↓ ⑧  CLI
# ─────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="KICE JSON → CoT jsonl + Gemini 검증")
    p.add_argument("--src", required=True, help="원본 KICE JSON 파일")
    p.add_argument("--out", required=True, help="출력 CoT jsonl 파일")
    p.add_argument("--gemini-out", help="(선택) Gemini 응답 jsonl 저장 경로")
    p.add_argument("--model", default="gemini-pro", help="Gemini 모델명 (기본: gemini-pro)")
    p.add_argument("--temp", type=float, default=0.3, help="Gemini temperature")
    return p.parse_args()

# ─────────────────────────────────────────────────────────────
# ↓↓↓ ⑨  main
# ─────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    rows = read_source(Path(args.src))

    # 1) jsonl 구축
    converted = [convert_row(r) for r in rows]
    dump_jsonl(converted, Path(args.out))
    print(f"✅  변환 완료: {len(converted):,} 문항 → {args.out}")

    # 2) Gemini 호출 (선택)
    if args.gemini_out:
        api_key = Path.home().joinpath(".config/gemini_api_key").read_text().strip() \
            if "GEMINI_API_KEY" not in dict(os.environ) else os.getenv("GEMINI_API_KEY")
        model = init_gemini(api_key, args.model)
        gemini_results = []

        for idx, conv in enumerate(converted, 1):
            reply = gemini_generate(model, conv["messages"], temp=args.temp)
            gemini_results.append({
                "id": conv["id"],
                "gemini_reply": reply,
            })
            if idx % 50 == 0:
                print(f"  …Gemini 진행 {idx}/{len(converted)}")
            time.sleep(0.1)  # QPS 완화

        dump_jsonl(gemini_results, Path(args.gemini_out))
        print(f"✅  Gemini 응답 저장: {args.gemini_out}")

if __name__ == "__main__":
    import os  # (늦게 import → 필요 시)
    main()