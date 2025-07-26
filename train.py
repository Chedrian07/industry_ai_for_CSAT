#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CUDA 환경에서 EXAONE-4.0-1.2B 모델 파인튜닝 스크립트
Transformers + LoRA 조합 사용
"""

import os
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any
import logging

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training
)
import numpy as np
from tqdm import tqdm

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatDataset(Dataset):
    """채팅 데이터셋 클래스"""
    
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 2048):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        messages = item.get("messages", [])
        text = self.format_chat_messages(messages)
        
        # 토크나이징
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt"
        )
        
        input_ids = encodings["input_ids"].squeeze()
        
        return {
            "input_ids": input_ids,
            "labels": input_ids.clone()  # 언어 모델링에서는 input과 label이 동일
        }
    
    def format_chat_messages(self, messages: List[Dict]) -> str:
        """채팅 메시지를 텍스트로 변환합니다."""
        formatted_text = ""
        
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            
            if role == "system":
                formatted_text += f"<|system|>\n{content}\n"
            elif role == "user":
                formatted_text += f"<|user|>\n{content}\n"
            elif role == "assistant":
                formatted_text += f"<|assistant|>\n{content}\n"
        
        formatted_text += "<|endoftext|>"
        return formatted_text

class ExaoneTrainer:
    """EXAONE 모델 훈련 클래스"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 시드 설정
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        
        logger.info(f"사용 디바이스: {self.device}")
        logger.info(f"CUDA 사용 가능: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"GPU 개수: {torch.cuda.device_count()}")
            logger.info(f"GPU 이름: {torch.cuda.get_device_name()}")
    
    def load_model_and_tokenizer(self):
        """모델과 토크나이저를 로드합니다."""
        logger.info(f"모델과 토크나이저 로딩 중: {self.args.model_name}")
        
        # 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.model_name,
            trust_remote_code=True,
            use_fast=False
        )
        
        # 패딩 토큰 설정
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # 모델 로드
        self.model = AutoModelForCausalLM.from_pretrained(
            self.args.model_name,
            torch_dtype=torch.float16 if self.args.use_fp16 else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
            use_cache=False  # 훈련 시 캐시 비활성화
        )
        
        # 모델을 훈련 모드로 설정
        self.model.train()
        
        logger.info("✅ 모델과 토크나이저 로딩 완료!")
        logger.info(f"모델 파라미터 수: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def setup_lora(self):
        """LoRA 설정을 적용합니다."""
        logger.info("LoRA 설정 적용 중...")
        
        # LoRA 설정
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.args.lora_rank,
            lora_alpha=self.args.lora_alpha,
            lora_dropout=self.args.lora_dropout,
            target_modules=[
                "q_proj",
                "k_proj", 
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj"
            ],
            bias="none",
        )
        
        # 모델을 LoRA로 준비
        if self.args.use_fp16:
            self.model = prepare_model_for_kbit_training(self.model)
        
        self.model = get_peft_model(self.model, lora_config)
        
        # 훈련 가능한 파라미터 수 출력
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        logger.info(f"훈련 가능한 파라미터: {trainable_params:,}")
        logger.info(f"전체 파라미터: {total_params:,}")
        logger.info(f"훈련 가능한 비율: {100 * trainable_params / total_params:.2f}%")
        logger.info("✅ LoRA 설정 완료!")
    
    def load_data(self):
        """데이터를 로드합니다."""
        logger.info("데이터셋 로딩 중...")
        
        # 훈련 데이터 로드
        train_data = self.load_jsonl_data(self.args.train_data_path)
        valid_data = self.load_jsonl_data(self.args.valid_data_path)
        
        logger.info(f"훈련 데이터: {len(train_data)}개")
        logger.info(f"검증 데이터: {len(valid_data)}개")
        
        if len(train_data) == 0:
            raise ValueError("훈련 데이터를 찾을 수 없습니다!")
        
        # 데이터셋 생성
        self.train_dataset = ChatDataset(train_data, self.tokenizer, self.args.max_sequence_length)
        self.valid_dataset = ChatDataset(valid_data, self.tokenizer, self.args.max_sequence_length) if valid_data else None
        
        logger.info("✅ 데이터셋 로딩 완료!")
    
    def load_jsonl_data(self, file_path: str) -> List[Dict]:
        """JSONL 파일을 로드합니다."""
        data = []
        if not os.path.exists(file_path):
            return data
            
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line.strip()))
        return data
    
    def create_data_collator(self):
        """데이터 콜레이터를 생성합니다."""
        return DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM이므로 False
            pad_to_multiple_of=8 if self.args.use_fp16 else None,
        )
    
    def train(self):
        """훈련을 실행합니다."""
        logger.info("훈련 시작!")
        
        # 모델과 토크나이저 로드
        self.load_model_and_tokenizer()
        
        # LoRA 설정
        if self.args.use_lora:
            self.setup_lora()
        
        # 데이터 로드
        self.load_data()
        
        # 데이터 콜레이터 생성
        data_collator = self.create_data_collator()
        
        # 훈련 인수 설정
        training_args = TrainingArguments(
            output_dir=self.args.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=self.args.num_epochs,
            per_device_train_batch_size=self.args.batch_size,
            per_device_eval_batch_size=self.args.batch_size,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            learning_rate=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            warmup_steps=self.args.warmup_steps,
            logging_steps=10,
            logging_dir=f"{self.args.output_dir}/logs",
            evaluation_strategy="epoch" if self.valid_dataset else "no",
            save_strategy="epoch",
            save_total_limit=3,
            load_best_model_at_end=True if self.valid_dataset else False,
            metric_for_best_model="eval_loss" if self.valid_dataset else None,
            greater_is_better=False,
            report_to=None,  # wandb 등 비활성화
            dataloader_pin_memory=True,
            gradient_checkpointing=self.args.gradient_checkpointing,
            fp16=self.args.use_fp16,
            dataloader_num_workers=4,
            remove_unused_columns=False,
            max_grad_norm=self.args.max_grad_norm,
        )
        
        # 트레이너 생성
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.valid_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        # 훈련 시작
        logger.info("🚀 훈련 시작!")
        trainer.train()
        
        # 최종 모델 저장
        final_output_dir = os.path.join(self.args.output_dir, "final_model")
        trainer.save_model(final_output_dir)
        self.tokenizer.save_pretrained(final_output_dir)
        
        logger.info(f"🎉 훈련 완료! 모델이 {final_output_dir}에 저장되었습니다.")

def main():
    parser = argparse.ArgumentParser(description="EXAONE-4.0-1.2B Fine-tuning with CUDA")
    
    # 모델 관련 설정
    parser.add_argument("--model_name", type=str, 
                       default="LGAI-EXAONE/EXAONE-4.0-1.2B",
                       help="모델 이름 또는 경로")
    
    # 데이터 관련 설정
    parser.add_argument("--train_data_path", type=str, 
                       default="./dataset/train.jsonl",
                       help="훈련 데이터 경로")
    parser.add_argument("--valid_data_path", type=str, 
                       default="./dataset/valid.jsonl",
                       help="검증 데이터 경로")
    
    # 훈련 관련 설정
    parser.add_argument("--batch_size", type=int, default=4,
                       help="배치 크기")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                       help="기울기 누적 스텝 수")
    parser.add_argument("--max_sequence_length", type=int, default=2048,
                       help="최대 시퀀스 길이")
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="훈련 에폭 수")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                       help="학습률")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="가중치 감쇠")
    parser.add_argument("--warmup_steps", type=int, default=100,
                       help="웜업 스텝 수")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                       help="기울기 클리핑 임계값")
    
    # LoRA 관련 설정
    parser.add_argument("--use_lora", action="store_true",
                       help="LoRA 사용 여부")
    parser.add_argument("--lora_rank", type=int, default=64,
                       help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16,
                       help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                       help="LoRA dropout")
    
    # 최적화 관련 설정
    parser.add_argument("--use_fp16", action="store_true",
                       help="FP16 혼합 정밀도 사용")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                       help="그래디언트 체크포인팅 사용 (메모리 절약)")
    
    # 출력 관련 설정
    parser.add_argument("--output_dir", type=str, default="./output",
                       help="출력 디렉토리")
    parser.add_argument("--seed", type=int, default=42,
                       help="랜덤 시드")
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 설정 출력
    logger.info("=" * 50)
    logger.info("EXAONE-4.0-1.2B CUDA Fine-tuning")
    logger.info("=" * 50)
    logger.info(f"모델: {args.model_name}")
    logger.info(f"배치 크기: {args.batch_size}")
    logger.info(f"기울기 누적: {args.gradient_accumulation_steps}")
    logger.info(f"유효 배치 크기: {args.batch_size * args.gradient_accumulation_steps}")
    logger.info(f"학습률: {args.learning_rate}")
    logger.info(f"에폭 수: {args.num_epochs}")
    logger.info(f"최대 시퀀스 길이: {args.max_sequence_length}")
    logger.info(f"LoRA 사용: {args.use_lora}")
    if args.use_lora:
        logger.info(f"LoRA rank: {args.lora_rank}")
        logger.info(f"LoRA alpha: {args.lora_alpha}")
        logger.info(f"LoRA dropout: {args.lora_dropout}")
    logger.info(f"FP16 사용: {args.use_fp16}")
    logger.info(f"그래디언트 체크포인팅: {args.gradient_checkpointing}")
    logger.info("=" * 50)
    
    # 훈련 실행
    trainer = ExaoneTrainer(args)
    trainer.train()

if __name__ == "__main__":
    main()
