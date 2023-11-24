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
from utils.score_func import *
import argparse

def main(args):
    
    model_name = args.model_name #'OpenOrca-Platypus2-13B' # 'orca_mini_v3_7b'
    model_path = f'/path/models/{model_name}'

    u_prompt, a_prompt = load_prompt_template(model_name, model_path, device="cuda:0")
    llm = LLM(model=model_path, tensor_parallel_size = 1)
    
    summ_path = args.data_path #'../data/summarization/train_summarization.tsv'
    summ_df = pd.read_csv(summ_path)
    
    source = summ_df['SRC'].tolist()
    summary = summ_df['HYP'].tolist()
    gt_score = summ_df['Score'].tolist()

    with open(args.template_path, 'r') as file:
        templates = json.load(file)

    ##################### Step1: Choose Initial Aspect #####################
    aspect_definition = defaultdict(list)
    if args.aspect_category == 'train':
        aspect_list = ['consistency', 'fluency', 'relevance', 'coherence']
        aspect_definition['consistency'] = load_definition(args.template_path, aspect='consistency', definition_type=args.definition)
        aspect_definition['fluency'] = load_definition(args.template_path, aspect='fluency', definition_type=args.definition)
        aspect_definition['relevance'] = load_definition(args.template_path, aspect='relevance', definition_type=args.definition)
        aspect_definition['coherence'] = load_definition(args.template_path, aspect='coherence', definition_type=args.definition)
    elif args.aspect_category == 'test':
        aspect_list = ['relevance', 'factuality', 'readability']
        aspect_definition['relevance'] = load_definition(args.template_path, aspect='relevance', definition_type=args.definition)
        aspect_definition['factuality'] = load_definition(args.template_path, aspect='factuality', definition_type=args.definition)
        aspect_definition['readability'] = load_definition(args.template_path, aspect='readability', definition_type=args.definition)
    elif args.aspect_category == 'all':
        aspect_list = ['consistency', 'fluency', 'relevance', 'coherence', 'factuality', 'readability']
        aspect_definition['consistency'] = load_definition(args.template_path, aspect='consistency', definition_type=args.definition)
        aspect_definition['fluency'] = load_definition(args.template_path, aspect='fluency', definition_type=args.definition)
        aspect_definition['relevance'] = load_definition(args.template_path, aspect='relevance', definition_type=args.definition)
        aspect_definition['coherence'] = load_definition(args.template_path, aspect='coherence', definition_type=args.definition)
        aspect_definition['factuality'] = load_definition(args.template_path, aspect='factuality', definition_type=args.definition)
        aspect_definition['readability'] = load_definition(args.template_path, aspect='readability', definition_type=args.definition)
    

    ##################### Step2: Scoring #####################   
    score_inputs = {}
    # simple output scoring
    for aspect in aspect_list:
        definition = aspect_definition[aspect]
        
        # load custom prompt
        score_prompt_list = []
        for i in range(len(source)):
            temp = make_prompt_type(templates=templates,
                             u_prompt=u_prompt,
                             a_prompt=a_prompt, 
                             aspect=aspect, 
                             definition=definition, 
                             source=source[i], 
                             summary=summary[i], 
                             prompting=args.definition, 
                             n_aspect=args.n_aspect, 
                             scoring=args.scoring, 
                             aspect_cate=args.aspect_category,
                             task_desc_type=args.task_description)
            if i == 0:
                print("prompt:", temp)
            score_prompt_list.append(temp)
        score_inputs[aspect] = score_prompt_list
    
    # scoring
    output_score = {}
    for aspect in tqdm(aspect_list):
        output_score[aspect] = {}
        score_input = score_inputs[aspect]
        
        if args.score_func == 'logprobs_sum':
            sampling_params = SamplingParams(temperature=args.score_temperature, max_tokens=args.max_token, n=args.score_num, stop=[u_prompt, a_prompt, '</s>'], logprobs=args.score_logprobs)
        else:
            sampling_params = SamplingParams(temperature=args.score_temperature, max_tokens=args.max_token, n=args.score_num, stop=[u_prompt, a_prompt, '</s>'], logprobs=None)
        score_output = generate_score(llm, score_input, sampling_params, score_func=args.score_func)
            
        if args.train=='True':
            correlation_output = correlation_result(score_output, gt_score)
            output_score[aspect]['correlation'] = correlation_output

        output_score[aspect]['score'] = score_output
        
    final_score_sum=0
    for aspect in aspect_list:
        final_score_sum += np.array(output_score[aspect]['score'])

    output_score['final_score'] = {}

    final_score_mean = final_score_sum / len(aspect_list)
    output_score['final_score']['scores'] = final_score_mean.tolist()
          
    if args.train=='True':
        output_score['final_score']['correlation'] = correlation_result(output_score['final_score']['scores'], gt_score)
    else:
        pd.DataFrame(data={'pred_score':output_score['final_score']['scores']}).to_csv(args.submit_result_path + "seg.scores", header=False, index=False)
        
    with open(os.path.join(args.final_score_output_path,'final_result.json'), 'w') as file:
        json.dump(output_score, file, indent=4)

                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # load path
    parser.add_argument('--model_name', type=str, default='OpenOrca-Platypus2-13B')
    parser.add_argument('--data_path', type=str, default='path/train_summarization.tsv')
    parser.add_argument('--template_path', type=str, default='prompt_template.json')
    
    # scoring
    parser.add_argument('--score_func', type=str, default='logprobs_sum', 
                        help='direct_generation')
    parser.add_argument('--score_temperature', type=float, default=0.7)
    parser.add_argument('--score_num', type=int, default=20)
    parser.add_argument('--score_logprobs', type=int, default=None)
    
    parser.add_argument('--train', type=str, default='True')
    parser.add_argument('--definition', type=str, default='summeval')
    parser.add_argument('--max_token', type=int, default=5)
    parser.add_argument('--aspect_category', type=str, default='train')
    parser.add_argument('--n_aspect', type=str, default='single_aspect')
    parser.add_argument('--scoring', type=str, default='five')
    
    parser.add_argument('--task_description', type=str, default=None)
    
    args = parser.parse_args()
    
    if args.score_func == 'logprobs_sum':
        args.final_score_output_path = os.path.join('path', args.score_func, args.model_name, f'temp_{args.score_temperature}', f'num{args.score_num}_token{args.max_token}_logprobs{args.score_logprobs}_nAspect{args.n_aspect}_range{args.scoring}_category{args.aspect_category}_task{args.task_description}')
    else:
        args.final_score_output_path = os.path.join('path', args.score_func, args.model_name, f'temp_{args.score_temperature}', f'num{args.score_num}_token{args.max_token}_logprobsX_nAspect{args.n_aspect}_range{args.scoring}_category{args.aspect_category}_task{args.task_description}')
    os.makedirs(args.final_score_output_path,exist_ok=True)
    
    main(args)
