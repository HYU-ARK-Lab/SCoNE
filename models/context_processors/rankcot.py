from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from models.context_processors.context_processor import ContextProcessor
import torch
from typing import List
from tqdm import tqdm


RANKCOT_PROMPT = "Passage:{passages}\nBased on these passages, answer the question below.\nQuestion:{question}\nLet's think step by step."


class RankCoTProcessor(ContextProcessor):

    def __init__(
        self,
        base_model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        adapter_name: str = "MignonMiyoung/RankCoT",
        batch_size: int = 4,
        max_passage_tokens: int = 1500,
        max_new_tokens: int = 512,
        top_k_passages: int = 5,
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.model = PeftModel.from_pretrained(base_model, adapter_name)
        self.model.eval()

        self.batch_size = batch_size
        self.max_passage_tokens = max_passage_tokens
        self.max_new_tokens = max_new_tokens
        self.top_k_passages = top_k_passages
        self.name = "rankcot"

    def _build_prompt(self, query: str, docs: List[str]) -> str:
        docs = docs[:self.top_k_passages]
        passages_str = "\n\n".join(
            [f"Document {i+1}:\n{doc}" for i, doc in enumerate(docs)]
        )
        # truncate passages to max_passage_tokens
        tokens = self.tokenizer.encode(passages_str)
        if len(tokens) > self.max_passage_tokens:
            passages_str = self.tokenizer.decode(tokens[:self.max_passage_tokens], skip_special_tokens=True)
        return RANKCOT_PROMPT.format(passages=passages_str, question=query)

    def _process(self, contexts: List[List[str]], queries: List[str]):
        prompts = [self._build_prompt(q, docs) for q, docs in zip(queries, contexts)]
        cot_outputs = []

        for i in tqdm(range(0, len(prompts), self.batch_size), desc="RankCoT processing"):
            batch_prompts = prompts[i: i + self.batch_size]
            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(self.device)

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    # temperature=1.0,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            prompt_len = inputs["input_ids"].size(1)
            generated = output_ids[:, prompt_len:]
            decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
            cot_outputs.extend(decoded)

        # return each CoT as a single-element list (one "document" per query)
        processed_contexts = [[f"Chain of Thought:\n{cot}"] for cot in cot_outputs]
        return processed_contexts, {}
