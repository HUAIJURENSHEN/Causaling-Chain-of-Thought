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
    solution = input_d['ground_truth_solution']
    response = input_d['response']
    prompt = template.format(problem=problem, solution=solution, response=response)
    messages = [{'role': 'user', 'content': prompt}]
    return messages

origin_model_dir = 'your_model_dir/Qwen2.5-7B-Instruct/Qwen/Qwen2___5-7B-Instruct'
llm = LLM(
    model=origin_model_dir, tokenizer=origin_model_dir,
    gpu_memory_utilization=0.75,
    tensor_parallel_size=2,
    enable_prefix_caching=True, 
    swap_space=16,
    max_num_seqs=20,
)
sampling_params = SamplingParams(temperature=1, top_p=0.9, n=1,
                                max_tokens=256, seed=42)
toker = AutoTokenizer.from_pretrained(origin_model_dir)
TEMPLATE = open('./templates/judge_template.txt').read().strip()

# ds_path = './ZR_result/Llama3.1-8B.jsonl'
# ds_path = './CoT_result/Llama3.1-8B.jsonl'
ds_path = './CaCoT_result/Llama3.1-8B.jsonl'

data = []
with open(ds_path, 'r') as file:
    for line in file:
        data.append(json.loads(line.strip()))

prompt_token_ids = []
for e in data:
    tokenized_input = prepare_input_boxed_query(TEMPLATE, e)
    tokenized_prompt = apply_chat_template(toker, tokenized_input)
    prompt_token_ids.append(tokenized_prompt)

generations_query = llm.generate(prompt_token_ids=prompt_token_ids, sampling_params=sampling_params)
resps_query = []
cnt = 0
for i in range(len(data)):
    generated = generations_query[i].outputs[0].text
    if generated == 'Yes':
        cnt += 1

print(ds_path)
print(cnt)
print(cnt / len(data))
