"""
EXAONE-4.0-1.2B Inference Test
Model: https://huggingface.co/LGAI-EXAONE/EXAONE-4.0-1.2B

Usage:
    python3 test_exaone_inference.py
    HF_TOKEN=hf_xxx python3 test_exaone_inference.py

NOTE: EXAONE-4.0-1.2B is a gated model. You must:
  1. Accept the license at https://huggingface.co/LGAI-EXAONE/EXAONE-4.0-1.2B
  2. Provide a valid HuggingFace token via HF_TOKEN env var or huggingface-cli login
"""

import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "LGAI-EXAONE/EXAONE-4.0-1.2B"

hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    print(
        "WARNING: HF_TOKEN not set. If this is a gated model, download will fail.\n"
        "  Set it with: export HF_TOKEN=hf_your_token_here\n"
    )

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

print(f"Device: {DEVICE}")
print(f"Dtype:  {DTYPE}")
print(f"Loading model: {MODEL_ID}\n")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    device_map="auto" if torch.cuda.is_available() else None,
    token=hf_token,
    trust_remote_code=True,
)
if DEVICE == "cpu":
    model = model.to(DEVICE)
model.eval()

print("Model loaded successfully.\n")
print("=" * 60)


def generate(prompt: str, max_new_tokens: int = 200) -> str:
    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    new_tokens = output_ids[0][input_ids.shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


TEST_CASES = [
    {
        "label": "한국어 질문",
        "prompt": "인공지능이란 무엇인가요? 간단하게 설명해주세요.",
    },
    {
        "label": "English question",
        "prompt": "What is the capital of South Korea? Answer briefly.",
    },
    {
        "label": "간단한 코드 생성",
        "prompt": "Python으로 피보나치 수열을 출력하는 함수를 작성해주세요.",
    },
]

for tc in TEST_CASES:
    print(f"[{tc['label']}]")
    print(f"Prompt : {tc['prompt']}")
    response = generate(tc["prompt"])
    print(f"Response: {response}")
    print("=" * 60)
