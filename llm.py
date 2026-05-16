"""
llm.py — LLM wrapper
提供文本生成（generate）和语义向量提取（embed）两个功能。
"""

import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


class SimpleLLM:
    def __init__(self, model_name: str = "Qwen/Qwen2-7B-Instruct", device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"Loading {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map=self.device,
            trust_remote_code=True,
        )
        self.model.eval()
        print("Model loaded!")

    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        """根据 prompt 生成文本。"""
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        )
        return response.strip()

    @torch.no_grad()
    def embed(self, text: str, max_length: int = 512) -> np.ndarray:
        """将文本编码为语义向量（最后隐层均值池化）。"""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).to(self.device)

        outputs = self.model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden = outputs.hidden_states[-1]   # (1, seq, hidden)
        vec = last_hidden.mean(dim=1).squeeze(0)  # (hidden,)
        return vec.float().cpu().numpy()
