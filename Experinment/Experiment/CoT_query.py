from torch.multiprocessing import set_start_method
from multiprocessing import Process, set_start_method

from transformers import BertTokenizer, BertModel
import torch
import argparse
import numpy as np
import torch
import json
from collections import Counter
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from datasets import load_dataset
import re
from modelscope.msdatasets import MsDataset

def apply_chat_template(toker, messages):
    input_prompt = toker.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return toker(input_prompt, add_special_tokens=False).input_ids

# generate messages
def prepare_input_boxed_query(template, input_d):
    problem = input_d['problem']
    prompt = template.format(problem=problem)
    messages = [{'role': 'user', 'content': prompt}]
    return messages

origin_model_dir = 'your_model_path/Qwen2.5-14B-Instruct/Qwen/Qwen2___5-14B-Instruct'

llm = LLM(
    model=origin_model_dir, tokenizer=origin_model_dir,
    gpu_memory_utilization=0.95,
    tensor_parallel_size=2,
    enable_prefix_caching=True, 
    swap_space=16,
    max_num_seqs=20,
)
sampling_params = SamplingParams(temperature=1, top_p=0.9, n=1,
                                max_tokens=4096, seed=42)
toker = AutoTokenizer.from_pretrained(origin_model_dir)
TEMPLATE = open('./templates/CoT.txt').read().strip()

ds_path = './Data_collection.jsonl'
data = []
with open(ds_path, 'r') as file:
    n = 0
    for line in file:
        data.append(json.loads(line.strip()))

prompt_token_ids = []
for e in data:
    tokenized_input = prepare_input_boxed_query(TEMPLATE, e)
    tokenized_prompt = apply_chat_template(toker, tokenized_input)
    prompt_token_ids.append(tokenized_prompt)

generations_query = llm.generate(prompt_token_ids=prompt_token_ids, sampling_params=sampling_params)
resps_query = []
for i in range(len(data)):
    d = {}
    generated = generations_query[i].outputs[0].text
    d['problem'] = data[i]['problem']
    d['ground_truth_solution'] = data[i]['ground_truth_solution']
    d['response'] = generated
    resps_query.append(d)

file_name = './CoT_result/Qwen2.5-14B.jsonl'
print(file_name)
with open(file_name, 'w', encoding='utf8') as file:
    for obj in resps_query:
        if "\n" in str(obj):
            obj = {k: v.replace("\n", "") if isinstance(v, str) else v 
                for k, v in obj.items()}
        line = json.dumps(obj, ensure_ascii=False)
        file.write(line + "\n")