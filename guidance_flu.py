import os

import sys
from utils.model_dict import *
import pandas as pd
import json
from tqdm import tqdm
import numpy as np
from vllm import LLM, SamplingParams

from tqdm import tqdm
from collections import defaultdict
from utils.prompt import *
from eval4nlp.sangmin.score_func import *
import argparse

import pandas as pd
import csv

import guidance
import pandas as pd
import torch
import torch.nn.functional as F

def weighted_sum(scores):
    # Convert scores to tensor
    scores_tensor = torch.tensor(list(scores.values()), dtype=torch.float32)
    
    # Compute softmax probabilities
    probabilities_tensor = F.softmax(scores_tensor, dim=0)
    
    # Compute weighted sum
    keys_tensor = torch.tensor([float(key) for key in scores.keys()], dtype=torch.float32)
    result = torch.dot(keys_tensor, probabilities_tensor)
    
    return result.item()

def parse_output(output):
    try:
        matched = re.search("^ ?([\d\.]+)", output)
        if (matched):
            try:
                score = float(matched.group(1))
            except:
                score = 0
        else:
            score = 0
    except:
        score = 0
    return score

def main(args):

    data_path = 'data_path'

    data = pd.read_csv(data_path)


    model_name = 'Platypus2-70B-Instruct-GPTQ' # orca_mini_v3_7b Platypus2-70B-Instruct-GPTQ
    model_path = f'/path/{model_name}'
    model, tokenizer, u_prompt, a_prompt = load_from_catalogue(model_name, model_path, 'cuda:0')

    g_model = guidance.llms.Transformers(
                model, tokenizer=tokenizer, trust_remote_code=True)

    guidance.llms.Transformers.cache.clear()
    guidance.llm = g_model

    if args.aspect_category == 'multi':
        
        flu = "Fluency:\nThis rating measures the quality of individual sentences, are they well-written and grammatically correct.\nConsider the quality of individual sentences."
            
        if args.score_func == 'sampling_sum':
            input_prompt = "{{u_prompt}}\nIn this task you will evaluate the quality of a summary written for a news article.\nTo correctly solve this task, follow these steps:\n\n1. Carefully read the news article, be aware of the information it contains.\n2. Read the proposed summary.\n3. Rate each summary on a scale from 1 (worst) to 5 (best) by its {{aspect}}.\n\n# Definition:\n{{definition}}\n----\nSource text: {{source}}\nSummary: {{summary}}\n\n{{a_prompt}}\n\nScore: {{gen 'score' n=20 temperature=1 max_tokens=5}}" 
            structure_program = guidance(input_prompt, llm=g_model, caching=False)
            
            con_score = []
            for i in tqdm(range(len(data))):
                zero_shot = structure_program(
                    u_prompt=u_prompt,
                    aspect="fluency",
                    definition=flu,
                    source=data['SRC'][i],
                    summary=data['HYP'][i],
                    a_prompt=a_prompt,
                    silent=True
                )
                sampling_parse_list = [parse_output(x) for x in zero_shot['score']]
                con_score.append(np.mean(np.array(sampling_parse_list)))

            
            df = pd.DataFrame({"flu":con_score})
        
    df.to_csv(f"/path/70b_{args.aspect_category}_{args.score_func}1_dev_flu.csv", index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--score_func', type=str, default='direct_generation')
    parser.add_argument('--aspect_category', type=str, default='multi')
    
    args = parser.parse_args()
    
    main(args)