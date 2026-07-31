'''
BERGEN
Copyright (c) 2024-present NAVER Corp.
CC BY-NC-SA 4.0 license
'''
import json
from pathlib import Path
import torch
import gc
from abc import ABC, abstractmethod
from modules.dataset import Tokenized_Sorted_Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from jinja2.exceptions import TemplateError
from functools import partial
import random
import numpy as np
import torch.nn.functional as F
from collections import Counter
import os
import hashlib
import pickle
from .neuron_strategies import NeuronStrategies

DS_NAME_MAP = {
    "hotpotqa": ("hotpotqa/hotpot_qa", "distractor")
}

class Generator(ABC):
    def __init__(self,
                 model_name: str = None,
                 batch_size: int = 1,
                 max_new_tokens: int = 1,
                 max_doc_len: int = 10**10,
                 max_length: int = None,
                 use_middle_truncation: bool = False):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.max_doc_len = max_doc_len
        self.max_length = max_length
        self.use_middle_truncation = use_middle_truncation

    @abstractmethod
    def generate(self, inp):
        pass
    
    @abstractmethod
    def collate_fn(self, inp):
        pass

    def pos_list2str(self, pos_list):
        return '@'.join([str(pos) for pos in pos_list])

    def pos_str2list(self, pos_str):
        return [int(pos) for pos in pos_str.split('@')]
    
    def scaled_input(self, minimum_activations, maximum_activations, batch_size, num_batch):
        """Attr을 위한 scaled input 생성"""
        num_points = batch_size * num_batch
        step = (maximum_activations - minimum_activations) / num_points
        res = torch.cat([torch.add(minimum_activations, step * i) for i in range(num_points)], dim=0)
        return res, step[0]
    
    def make_wo_instruction(self, instruction):
        """
        instruction에서 Background부터 Question 이전까지의 내용을 삭제하는 함수
        """

        if "Background" in instruction and "Question" in instruction:
            start = instruction.find("Background")
            end = instruction.find("Question")

            before_bg = instruction[:start]
            from_question = instruction[end:]
            wo_instruction = before_bg + from_question
        
        else:
            wo_instruction = instruction

        wo_instruction = wo_instruction.replace(
            "Your task is to extract relevant information from provided documents and to answer to questions as briefly as possible",
            "Answer the questions as briefly as possible"  
        )
        return wo_instruction.strip()

    def _make_ds_prompt(self, sample):
        """Build w/wo context prompts from a raw HF dataset sample (hotpotqa)."""
        system_prompt = "Your task is to extract relevant information from provided documents and to answer to questions as briefly as possible"
        titles = sample['context']['title']
        sentences = sample['context']['sentences']
        docs = []
        for i, (title, sents) in enumerate(zip(titles, sentences)):
            doc_text = ''.join(sents)
            docs.append(f"Document {i+1}: {title} {doc_text}")
        context_str = '\n'.join(docs)
        question = sample['question']
        user_content = f"Background: {context_str}\nQuestion: {question}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        prompt_with_context = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        prompt_without_context = self.make_wo_instruction(prompt_with_context)
        return prompt_without_context, prompt_with_context

    def get_context_attr(self, idx, prompt_without_context, prompt_with_context, answer_obj):
        """
        Integrated-gradients context-aware neuron attribution, adapted from IRCAN:
        Shi et al., "IRCAN: Mitigating Knowledge Conflicts in LLM Generation via
        Identifying and Reweighting Context-Aware Neurons", NeurIPS 2024.
        https://github.com/danshi777/IRCAN
        """
        device = self.model.device
        class Args:
            batch_size = 20
            num_batch = 1
            batch_size_per_inference = 2

        args = Args()

        tokens = self.tokenizer.tokenize(answer_obj)
        gold_label_id = self.tokenizer.convert_tokens_to_ids(tokens[0])
        
        wo_tokenized_inputs = self.tokenizer(prompt_without_context, return_tensors="pt")
        wo_input_ids = wo_tokenized_inputs["input_ids"].to(device)
        wo_attention_mask = wo_tokenized_inputs["attention_mask"].to(device)
        
        w_tokenized_inputs = self.tokenizer(prompt_with_context, return_tensors="pt")
        w_input_ids = w_tokenized_inputs["input_ids"].to(device)
        w_attention_mask = w_tokenized_inputs["attention_mask"].to(device)

        # record results
        res_dict = {
            'idx': idx,
            'wo_all_ffn_activations': [],
            'w_all_ffn_activations': [],
            'all_attr_gold': [],
        }

        for tgt_layer in range(self.model.model.config.num_hidden_layers):
            wo_ffn_activations_dict = dict()
            def wo_forward_hook_fn(module, inp, outp):
                wo_ffn_activations_dict['input'] = inp[0]

            w_ffn_activations_dict = dict()
            def w_forward_hook_fn(module, inp, outp):
                w_ffn_activations_dict['input'] = inp[0]

            # get activations: no context
            wo_hook = self.model.model.layers[tgt_layer].mlp.down_proj.register_forward_hook(wo_forward_hook_fn)
            with torch.no_grad():
                wo_outputs = self.model(input_ids=wo_input_ids, attention_mask=wo_attention_mask)
            wo_ffn_activations = wo_ffn_activations_dict['input']
            wo_ffn_activations = wo_ffn_activations[:, -1, :]
            wo_hook.remove()
            del wo_outputs

            # get activations: with context
            w_hook = self.model.model.layers[tgt_layer].mlp.down_proj.register_forward_hook(w_forward_hook_fn)
            with torch.no_grad():
                w_outputs = self.model(input_ids=w_input_ids, attention_mask=w_attention_mask)
            w_ffn_activations = w_ffn_activations_dict['input']
            w_ffn_activations = w_ffn_activations[:, -1, :]
            w_hook.remove()
            del w_outputs

            wo_ffn_activations.requires_grad_(True)
            w_ffn_activations.requires_grad_(True)

            scaled_activations, activations_step = self.scaled_input(wo_ffn_activations, w_ffn_activations, args.batch_size, args.num_batch)
            scaled_activations.requires_grad_(True)

            ig_gold = None

            for batch_idx in range(args.num_batch):
                all_grads = None
                for i in range(0, args.batch_size, args.batch_size_per_inference):
                    batch_activations = scaled_activations[i: i + args.batch_size_per_inference]

                    batched_w_input_ids = w_input_ids.repeat(args.batch_size_per_inference, 1)
                    batched_w_attention_mask = w_attention_mask.repeat(args.batch_size_per_inference, 1)

                    def i_forward_hook_change_fn(module, inp):
                        inp0 = inp[0]
                        inp0[:, -1, :] = batch_activations
                        return (inp0,)

                    change_hook = self.model.model.layers[tgt_layer].mlp.down_proj.register_forward_pre_hook(i_forward_hook_change_fn)
                    outputs = self.model(input_ids=batched_w_input_ids, attention_mask=batched_w_attention_mask)

                    tgt_logits = outputs.logits[:, -1, :]
                    tgt_probs = F.softmax(tgt_logits, dim=1)
                    del tgt_logits, outputs

                    is_last_minibatch = (i + args.batch_size_per_inference >= args.batch_size)
                    grads_i = torch.autograd.grad(
                        torch.unbind(tgt_probs[:, gold_label_id]),
                        w_ffn_activations,
                        retain_graph=not is_last_minibatch
                    )
                    del tgt_probs
                    change_hook.remove()

                    all_grads = grads_i[0].detach() if all_grads is None else torch.cat((all_grads, grads_i[0].detach()), dim=0)
                    del grads_i

                grad = all_grads.sum(dim=0)
                ig_gold = grad if ig_gold is None else torch.add(ig_gold, grad)

            ig_gold = ig_gold * activations_step
            res_dict['wo_all_ffn_activations'].append(wo_ffn_activations.detach().cpu().squeeze().tolist())
            res_dict['w_all_ffn_activations'].append(w_ffn_activations.detach().cpu().squeeze().tolist())
            res_dict['all_attr_gold'].append(ig_gold.detach().cpu().tolist())

            del wo_ffn_activations, w_ffn_activations, scaled_activations, ig_gold, all_grads, grad
            torch.cuda.empty_cache()

        return res_dict
    
    def convert_to_triplet_ig(self, ig_list):
        ig_triplet = []
        ig = np.array(ig_list) # (layer_num, ffn_size)
        max_ig = ig.max()  # maximum attribution score
        for i in range(ig.shape[0]):
            for j in range(ig.shape[1]):
                if ig[i][j] >= max_ig * 0.1: 
                    ig_triplet.append([i, j, ig[i][j]])
        return ig_triplet
    
    def convert_all_neurons_attribution_score_to_triplet_ig(self, ig_list):
        ig_triplet = []
        ig = np.array(ig_list) # (layer_num, ffn_size)
        for i in range(ig.shape[0]):
            for j in range(ig.shape[1]):
                ig_triplet.append([i, j, ig[i][j]])
        return ig_triplet    

    def final_evaluate(self, eval_dataset, experiment_mode, train_dataset_name, use_attr, original_use_cad, save_result,
                       enhance_strength=5.0, top_k=50):
        w_responses, w_instructions = list(), list()

        query_ids, queries, labels, ranking_labels = list(), list(), list(), list()
        results = list()
        self.use_cad = original_use_cad

        with torch.no_grad():
            if self.tokenizer:
                enhanced_dataset = Tokenized_Sorted_Dataset(eval_dataset, self, training=False)
                enhanced_dataloader = DataLoader(enhanced_dataset, batch_size=self.batch_size, 
                                            collate_fn=partial(self.collate_fn, eval=True), num_workers=4)
            else:
                enhanced_dataloader = DataLoader(eval_dataset, batch_size=self.batch_size, 
                                            collate_fn=partial(self.collate_fn, eval=True), num_workers=4)
                
            for data_dict in tqdm(enhanced_dataloader, desc='Final Evaluating'):
                id_ = data_dict['q_id']
                w_instruction = data_dict['instruction']

                query_ids += id_
                label = data_dict['label']
                labels += label
                queries += data_dict['query']
                ranking_labels += data_dict['ranking_label']

                w_instructions += w_instruction

                
                if self.use_cad:
                    wo_instruction = []
                    for query in data_dict['query']:
                        messages = [
                            {"role": "system", "content": "You are a helpful assistant. Answer the questions as briefly as possible."},
                            {"role": "user", "content": f"Question: {query}"},
                        ]
                        direction = self.tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True
                        )
                        wo_instruction.append(direction)
                    
                    wo_tokenized = self.tokenizer(
                        wo_instruction,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=self.max_length if hasattr(self, 'max_length') and self.max_length else None
                    )
                    w_generated_response = self.generate(
                        data_dict['model_input'],
                        wo_tokenized
                    )
                else:
                    w_generated_response = self.generate(
                        data_dict['model_input']
                    )

                w_responses += w_generated_response
                for i in range(len(id_)):
                    results.append({
                        'q_id': id_[i],
                        'model_input': w_instruction[i],
                        'response': w_generated_response[i],
                        'label': label[i]
                    })  

                torch.cuda.empty_cache()
                gc.collect()

        if save_result:
            output_dir = Path("response_logs")
            output_dir.mkdir(exist_ok=True)

            # results 저장
            output_file = output_dir / f"{train_dataset_name}_use_attr_{use_attr}_use_cad_{self.use_cad}_no_context_{self.no_context}_exp_{experiment_mode}_strength{enhance_strength}_topk{top_k}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
        return query_ids, queries, w_instructions, w_responses, labels, ranking_labels

        
    def eval(self, eval_dataset,
            train_dataset=None,
            experiment_mode='original',
            train_dataset_name=None,
            enhance_strength=5.0,
            top_k=50,
            top_n=50,
            save_result=False,
            attr_ds_name=None,
            num_attr_samples=100,
            retriever_name=None,
            cross_ds_name=None,
            neuron_selection_mode='first',
            random_seed=42,
            neuron_json_path=None):
        """
        eval_dataset: 평가할 데이터셋 (dev/test)
        train_dataset: CN 발견용 데이터셋 (train, optional)
        """
        use_attr = (train_dataset is not None) or (attr_ds_name is not None)
        if train_dataset_name is None and attr_ds_name is not None:
            train_dataset_name = f"{attr_ds_name}_n{num_attr_samples}"
        if train_dataset_name is not None:
            if neuron_selection_mode == 'random':
                train_dataset_name = f"{train_dataset_name}_random_seed{random_seed}"
            elif neuron_selection_mode == 'last':
                train_dataset_name = f"{train_dataset_name}_last"
            else:
                train_dataset_name = f"{train_dataset_name}_first"
        if experiment_mode == 'exp_cross':
            use_attr = True
        original_use_cad = self.use_cad
        self.use_cad = False

        if use_attr:
            attr_results = list()
            all_neuron_stats = {}

            if experiment_mode != 'exp_cross':
                wo_instructions, w_instructions, query_ids, labels = list(), list(), list(), list()

                # HuggingFace dataset을 직접 로드하여 attr용 프롬프트 생성
                from datasets import load_dataset as hf_load_dataset
                ds_hf_name, ds_subset = DS_NAME_MAP[attr_ds_name]
                ds = hf_load_dataset(ds_hf_name, ds_subset) if ds_subset else hf_load_dataset(ds_hf_name)
                train_ds = ds['train']
                total = len(train_ds)
                n = min(num_attr_samples, total)
                if neuron_selection_mode == 'random':
                    rng = random.Random(random_seed)
                    hf_sample_indices = sorted(rng.sample(range(total), n))
                elif neuron_selection_mode == 'last':
                    hf_sample_indices = list(range(max(0, total - n), total))
                else:
                    hf_sample_indices = list(range(n))
                # [traversal-seed ablation] 같은 100개를 그대로 두고 '가져온 순서'만 seed로 섞음.
                _trav_seed = os.environ.get("TRAVERSAL_SEED")
                if _trav_seed is not None:
                    random.Random(int(_trav_seed)).shuffle(hf_sample_indices)

                for i in tqdm(hf_sample_indices, desc='Building prompts from ds (Attr)'):
                    sample = train_ds[i]
                    wo_prompt, w_prompt = self._make_ds_prompt(sample)
                    query_ids.append(sample.get('id', i))
                    labels.append([sample['answer']])
                    wo_instructions.append(wo_prompt)
                    w_instructions.append(w_prompt)
                    torch.cuda.empty_cache()
                    gc.collect()
                attr_indices = list(range(len(wo_instructions)))

                # Attr 계산
                for i in tqdm(attr_indices, desc="Processing Attr"):
                    try:
                        res_dict = self.get_context_attr(
                            idx=query_ids[i],
                            prompt_without_context=wo_instructions[i],
                            prompt_with_context=w_instructions[i],
                            answer_obj=labels[i][0]
                        )
                        for layer_idx in range(len(res_dict['all_attr_gold'])):
                            for neuron_idx, score in enumerate(res_dict['all_attr_gold'][layer_idx]):
                                key = (layer_idx, neuron_idx)
                                if key not in all_neuron_stats:
                                    all_neuron_stats[key] = []
                                all_neuron_stats[key].append(score)

                        res_dict["all_attr_gold"] = self.convert_to_triplet_ig(res_dict["all_attr_gold"])

                        attr_results.append(res_dict)

                    except Exception as e:
                        print(f"Error processing sample {query_ids[i] if i < len(query_ids) else i}: {e}")

            print(f"\nSelecting neurons for experiment: {experiment_mode}, {enhance_strength}")

            ## Experiment Modes 선택하기
            strategy_map = {
                'ircan': lambda: NeuronStrategies.counter_attribution_desc(attr_results, top_k, train_dataset_name, model_name=self.model_name),
                'scone': lambda: NeuronStrategies.high_attr_high_var_counter_enhance(attr_results, all_neuron_stats, top_k=top_k, train_dataset_name=train_dataset_name, model_name=self.model_name, top_n=top_n),
                'scone_attronly': lambda: NeuronStrategies.high_attr_high_var_counter_enhance(attr_results, all_neuron_stats, top_k=top_k, train_dataset_name=train_dataset_name, model_name=self.model_name, top_n=top_n, select_mode='attr_only'),
                'scone_varonly': lambda: NeuronStrategies.high_attr_high_var_counter_enhance(attr_results, all_neuron_stats, top_k=top_k, train_dataset_name=train_dataset_name, model_name=self.model_name, top_n=top_n, select_mode='var_only'),
                'scone_var': lambda: NeuronStrategies.high_attr_high_dispersion_counter_enhance(attr_results, all_neuron_stats, top_k=top_k, train_dataset_name=train_dataset_name, model_name=self.model_name, top_n=top_n, var_mode='var'),
                'scone_std': lambda: NeuronStrategies.high_attr_high_dispersion_counter_enhance(attr_results, all_neuron_stats, top_k=top_k, train_dataset_name=train_dataset_name, model_name=self.model_name, top_n=top_n, var_mode='std'),
                'scone_mad': lambda: NeuronStrategies.high_attr_high_dispersion_counter_enhance(attr_results, all_neuron_stats, top_k=top_k, train_dataset_name=train_dataset_name, model_name=self.model_name, top_n=top_n, var_mode='mad'),
                'scone_w1': lambda: NeuronStrategies.high_attr_high_var_counter_enhance_window(attr_results, all_neuron_stats, top_k=top_k, train_dataset_name=train_dataset_name, model_name=self.model_name, window_size=1, top_n=top_n),
                'scone_w2': lambda: NeuronStrategies.high_attr_high_var_counter_enhance_window(attr_results, all_neuron_stats, top_k=top_k, train_dataset_name=train_dataset_name, model_name=self.model_name, window_size=2, top_n=top_n),
                'scone_w5': lambda: NeuronStrategies.high_attr_high_var_counter_enhance_window(attr_results, all_neuron_stats, top_k=top_k, train_dataset_name=train_dataset_name, model_name=self.model_name, window_size=5, top_n=top_n),
                'scone_w10': lambda: NeuronStrategies.high_attr_high_var_counter_enhance_window(attr_results, all_neuron_stats, top_k=top_k, train_dataset_name=train_dataset_name, model_name=self.model_name, window_size=10, top_n=top_n),
                'exp_cross': lambda: NeuronStrategies.cross_dataset_intersection_enhance(train_dataset_name, cross_ds_name, top_k, neuron_json=neuron_json_path),
            }
            
            
            if experiment_mode in strategy_map:
                cns = strategy_map[experiment_mode]()

            if not cns:
                print("[WARNING] cns가 비어있습니다. 뉴런 강화 없이 진행합니다.")
            else:
                for layer, pos in cns:
                    with torch.no_grad():
                        self.model.model.layers[layer].mlp.down_proj.weight[:, pos] *= enhance_strength

        return self.final_evaluate(
            eval_dataset,
            experiment_mode,
            train_dataset_name,
            use_attr,
            original_use_cad,
            save_result,
            enhance_strength=enhance_strength,
            top_k=top_k,
        )

    def get_response(self):
        """
        This replaces the 'generation_prompt' in case the generator does not have a chat_template.
        It's used to prompt and also to identify the label positions to mask prompt in training.
        """
        return '\nResponse:\n'

    def get_response_template_ids(self):
        response_template =  self.get_response()
        return self.tokenizer.encode(response_template, add_special_tokens=False)
    
    def compile_prompt(self, system_prompt: str, user_prompt: str, question: str, docs: str = None, label: str = None):
            """
            Applying the chat template if it exists:
            NB: seemingly unused args are used in the 'eval' call.
            NB: if the label is not None, we assume training=True and then the answer of the model is added to the full prompt
            This method returns a tuple consisting of:
            - the final prompt
            - if a label is provided, the position of the first label index within the tokenized sequence (for masking in training)
            """
            # the prompt should finish with a generation prompt if we are in 'eval' mode i.e. when there is no label
            # NB: the generation prompt is empty (automatically included in the template rather) for llama/mistral/solar at least
            add_generation_prompt = (label is None) 
            
            label_start_index = None
            if self.tokenizer.chat_template is None:
                user_prompt_with_values = eval(user_prompt).replace(':\ ', ': ')
                # we add the 'reponse incitation' to non chat template
                prompt = f"{system_prompt}\n{user_prompt_with_values}" + self.get_response()
                if label is not None:
                    # Compute prompt size in tokens without labels.
                    label_start_index = len(self.tokenizer(prompt, add_special_tokens=False)['input_ids'])
                    prompt += label + self.tokenizer.eos_token

            else:        
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": eval(user_prompt).replace(':\ ', ': ')}
                ]
                try:
                    # Handle the label
                    if label is not None:
                        # Compute the prompt without label, to know its length and hence where to mask in training
                        label_start_index = len(self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, add_special_tokens=False))
                        messages.append({"role": "assistant", "content": label})
                    
                    prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=add_generation_prompt, tokenize=False)

                except TemplateError as e:
                    if "System role not supported" in str(e):
                        messages = [{"role": "user", "content": messages[0]['content'] + '\n' + messages[1]['content']}]

                        if label is not None:
                            # Compute the prompt without label, to know its length and hence where to mask in training  
                            label_start_index = len(self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, add_special_tokens=False))
                            messages.append({"role": "assistant", "content": label})

                        prompt = self.tokenizer.apply_chat_template(messages,  add_generation_prompt=add_generation_prompt, tokenize=False)
                    else:
                        raise e
            
            
            if label is not None:
                assert label_start_index is not None # check we did find the prompt length
                if not prompt.endswith(self.tokenizer.eos_token):
                    prompt += self.tokenizer.eos_token # most models have this already, but not gemma-2b !
                
            return prompt, label_start_index

    def middle_truncation(self, docs):
        """
        Truncate documents by removing the middle section while preserving both the beginning and end.
        Args:
            docs (str): The document text to truncate
                
        Returns:
            str: The truncated document text
        """
        if docs is None or self.max_length is None or not hasattr(self, 'tokenizer'):
            return docs
        
        tokenized_docs = self.tokenizer(docs, truncation=False, return_tensors="pt")['input_ids'][0]
        docs_length = len(tokenized_docs)
        
        truncation_threshold = self.max_length - 128
        assert truncation_threshold >= 0, "Truncation threshold must be non-negative. Check max_length value."
        
        if docs_length > truncation_threshold:
            half = int(truncation_threshold / 2)
            
            first_half = tokenized_docs[:half]
            second_half = tokenized_docs[-half:]
            
            first_half_text = self.tokenizer.decode(first_half, skip_special_tokens=True)
            second_half_text = self.tokenizer.decode(second_half, skip_special_tokens=True)
            docs = first_half_text + second_half_text

        return docs


    def format_instruction(self, sample: dict, eval: bool = True) -> (str, int):
        """
        Makes the actual prompt from the prompt template and the model chat template
        Also return start index of the label in that prompt, if eval=True and a label is provided, None otherwise.
        If eval=True, then no label is added to the prompt.
        If eval=False, then the label is added to the prompt, for training (teacher forcing)
        """
        question = sample['query']
        label = None
        if not eval:
            label = (sample['label'] if isinstance(sample['label'], str) else random.choice(sample['label']))
            assert label is not None
        if 'doc' in sample:
            # We have retrieved documents:
            docs = ''
            input_docs = sample['doc']
            input_docs = [doc for doc in input_docs if len(doc.strip()) > 0]
            for i, doc in enumerate(input_docs):
                doc = ' '.join(doc.split()[:self.max_doc_len])
                docs += f"Document {i+1}: {doc}\n"
            if self.use_middle_truncation:
                docs = self.middle_truncation(docs)
            return self.compile_prompt(self.prompt.system, self.prompt.user, question, docs, label=label)
        else:
            # We have no retrieved documents: switch to no doc prompt
            return self.compile_prompt(self.prompt.system_without_docs, self.prompt.user_without_docs, question, label=label)
