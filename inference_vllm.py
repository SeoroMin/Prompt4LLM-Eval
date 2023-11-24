import os
#os.environ["CUDA_VISIBLE_DEVICES"]= "0"

# import sys
from model_dict import *
from inference import *
import pandas as pd
import guidance
import json
from tqdm import tqdm
# import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np
from vllm import LLM, SamplingParams
# import scipy.stats
from tqdm import tqdm
from collections import defaultdict
from prompt import *
from utils import *

def main(args):
    aspect_names = '_'.join(args.aspect_list)
    print(aspect_names)
    default_dir = os.path.join(args.score_output_path,args.template+'_template',aspect_names, args.exp_name)
    os.makedirs(default_dir, exist_ok=True)

    model_name = args.model_name #'OpenOrca-Platypus2-13B' # 'orca_mini_v3_7b'
    model_path = f'/code/SharedTask2023/models/{model_name}'
    
    summ_path = args.data_path #'../data/summarization/train_summarization.tsv'
    # summ_df = pd.read_csv(summ_path, delimiter="\t", quoting=csv.QUOTE_NONE)
    summ_df = pd.read_csv(summ_path)
    
    if args.train=='True':

        data_type = 'train'
        source = summ_df['SRC'].tolist()
        summary = summ_df['HYP'].tolist()
        gt_score = summ_df['Score'].tolist()

    else:
        data_type = 'test'

        source = summ_df['SRC'].tolist()
        summary = summ_df['TGT'].tolist()
    # source = source[:5]
    # summary = summary[:5]
    # gt_score = gt_score[:5]

    with open(args.template_path, 'r') as file:
        templates = json.load(file)

    ##################### Step1: Choose Initial Aspect #####################
    #aspect_list = ['consistency', 'fluency', 'relevance', 'coherence', 'readability', 'factuality', 'informativeness']
    # aspect_list = ['consistency', 'fluency', 'relevance', 'coherence', 'r']
    aspect_list = args.aspect_list
    aspect_definition = defaultdict(list)

    # for aspect in aspect_list:
    #     aspect_definition[aspect] = load_definition(args.template_path, aspect=aspect, definition_type=args.definition)

    
    generated_def_mapper = {'summeval':'summeval_new',
                            'geval' :'geval_new',
                            'gptscore':'gptscore_new',
                            'summeval_new':'summeval_new',
                            'geval_new' :'geval_new',
                            'gptscore_new':'gptscore_new',
                        'gptscore_new_no_ref':'gptscore_new_no_ref',
                        'gptscore_new_summeval_ref' :'gptscore_new_summeval_ref',
                        'gptscore_new_geval_ref':'gptscore_new_geval_ref',
                            }    
      
    generated_def_mapper_2 = {'summeval':'summeval_new',
                        'geval' :'geval_new',
                        'gptscore':'gptscore',
                        'summeval_new':'summeval_new',
                        'geval_new' :'geval_new',
                        'gptscore_new':'gptscore_new',
                        'gptscore_new_no_ref':'summeval_new',
                        'gptscore_new_summeval_ref' :'geval_new',
                        'gptscore_new_geval_ref':'gptscore_new',
                        }    
    
    aspect_definition['consistency'] = load_definition(args.template_path, aspect='consistency', definition_type=args.definition)
    aspect_definition['fluency'] = load_definition(args.template_path, aspect='fluency', definition_type=args.definition)
    aspect_definition['relevance'] = load_definition(args.template_path, aspect='relevance', definition_type=args.definition)
    aspect_definition['coherence'] = load_definition(args.template_path, aspect='coherence', definition_type=args.definition)
    aspect_definition['factuality'] = load_definition(args.template_path, aspect='factuality', definition_type=generated_def_mapper[args.definition])
    aspect_definition['informativeness'] = load_definition(args.template_path, aspect='informativeness', definition_type=generated_def_mapper[args.definition])
    aspect_definition['readability'] = load_definition(args.template_path, aspect='readability', definition_type=generated_def_mapper_2[args.definition])
    
    device = 'cuda:0'#+args.device

    if args.module=='vllm':
        u_prompt, a_prompt = load_prompt_template(model_name, model_path, device=device)
        model = LLM(model=model_path, tensor_parallel_size = 1)

    elif args.module=='guidance':
        model, tokenizer, u_prompt, a_prompt = load_from_catalogue(model_name, model_path, device)
        tokenizer.pad_token = '[PAD]'
        model = guidance.llms.Transformers(
                    model, tokenizer=tokenizer, trust_remote_code=True,
        )
        guidance.llms.Transformers.cache.clear()
        guidance.llm = model

    # Remove samples that repeat redundant words or phrases
    if args.rm_redundancy=='True':

        redund_prompt = rm_redun_prompts(u_prompt, summary, a_prompt)
        sampling_params_redund = SamplingParams(temperature=0.0, max_tokens=100, n=1, 
                                        stop=[u_prompt, a_prompt, '</s>'], 
                                        )
        
        result, explanation_list = remove_redun(redund_prompt, model, sampling_params_redund)

        no_list = [i for i in range(len(result)) if result[i] == 'Yes']
        no_list.sort()

    ##################### Step2: Scoring #####################   
    options = [str(i) for i in range(1, 6)]

    score_inputs = {}
    # simple output scoring
    if args.template=='baseline':
        aspect_list = ['single_aspect']

    for aspect in aspect_list:
        
        definition = aspect_definition[aspect]
        evaluation_step = templates['evaluation_step'][aspect]['geval']
        
        if args.example=='True':
            example = templates['experiment']['few_shot'][aspect][f'{args.example_score}']
            example_source = example['src']
            example_summary = example['hyp']
            example_score = example['score']
            example_rationale = example['explain']
        
        
        score_prompt_list = []

        if args.module=='vllm':
            if args.template=='summeval':
                for i in range(len(source)):
                    prompt = evaluation_prompt_summeval_base(u_prompt, a_prompt, 
                                                            source[i], summary[i], 
                                                            aspect=aspect, 
                                                            definition=definition,
                                                            data_type=data_type,
                                                            start = args.start,
                                                            end = args.end
                                                            )
                    score_prompt_list.append(prompt)
                
            elif args.template == 'baseline':
                for i in range(len(source)):
                    prompt = evaluation_prompt_baseline(u_prompt, a_prompt, source[i], summary[i], module = 'vllm')
                    
                    score_prompt_list.append(prompt)
            
            elif args.template == 'geval':
                for i in range(len(source)):
                    prompt = evaluation_prompt_geval_base(u_prompt, a_prompt, source[i], summary[i], 
                                                          aspect, definition, evaluation_step, data_type,
                                 start = 1, end = 5)
                    
                    score_prompt_list.append(prompt)
                    
            elif args.template == 'rationale':
                if args.example=='True':
                    for i in range(len(source)):
                        prompt = rationale_prompt_few(u_prompt, a_prompt, source[i], summary[i], 
                                                            aspect, definition,
                                                            example_source, example_summary, example_score, example_rationale)
                        score_prompt_list.append(prompt)
                
                else: 
                    for i in range(len(source)):

                        prompt = rationale_prompt_zero(u_prompt, a_prompt, source[i], summary[i],
                                                       aspect, definition, start = args.start, end = args.end)
                        score_prompt_list.append(prompt)
        
        elif args.module=='guidance':
            if args.template=='baseline':
                score_prompt_list = evaluation_prompt_baseline(None, None, None, None, module = 'guidance')
            elif args.template=='summeval':
                score_prompt_list = evaluation_prompt_summeval_guidance

        score_inputs[aspect] = score_prompt_list

    with open(os.path.join(default_dir,'summarization.json.prompt'), 'w') as file:
        json.dump(score_prompt_list, file, indent=4)

    # scoring
    output_score = {}
    for aspect in tqdm(aspect_list):
        score_output = []

        output_score[aspect] = {}
        score_input = score_inputs[aspect]

        if args.module=='vllm':

            sampling_params = SamplingParams(temperature=args.score_temperature, max_tokens=512, n=args.score_num, stop=[u_prompt, a_prompt, '</s>', '----', '\n\nSource', '\n\nThe', '\n\nScore'], logprobs=args.score_logprobs)
            
            if args.explanation=='True':
                score_output, explain_output = generate_score(model, score_input, sampling_params, score_func=args.score_func, explanation=True)
                
            else: 
                score_output = generate_score(model, score_input, sampling_params, score_func=args.score_func)
            
        elif args.module=='guidance':
            for i in tqdm(range(len(source))):

                structure_program = guidance(score_input, llm=model)

                res = structure_program(
                    aspect = aspect,
                    definition= aspect_definition[aspect],
                    source = source[i],
                    summary = summary[i],
                    u_prompt = u_prompt,
                    a_prompt = a_prompt,
                    options = options
                )

                res()
                torch.cuda.empty_cache()

                if args.template == 'baseline':
                    score = res['score']
                    try:
                        score = float(score)
                    except:
                        score = 0.0
                        
                    score_output.append(score)    

                else:
                    scores = res['logprobs']
                    score = weighted_sum(scores)
                    score_output.append(score)        
                    

        if args.train=='True':
            correlation_output = correlation_result(score_output, gt_score)
            output_score[aspect]['correlation'] = correlation_output

        if args.explanation=='True':
            output_score[aspect]['explain'] = explain_output
        
        output_score[aspect]['score'] = score_output

    final_score_sum=0
    for aspect in aspect_list:
        final_score_sum += np.array(output_score[aspect]['score'], dtype=np.float32)
    
    output_score['final_score'] = {}

    final_score_mean = final_score_sum / len(aspect_list)

    if args.rm_redundancy=='True':
        final_score_mean[no_list] = 0.0
        with open(os.path.join(default_dir,'redundancy_check.json'), 'w') as file:

            save_redun = {'explanation_list':explanation_list,
                            'Redundant Index' : no_list
                            }
            json.dump(save_redun, file, indent=4)

    output_score['final_score']['scores'] = final_score_mean.tolist()
    
    if args.binning=='True':
        output_score['final_score']['scores'] = binning(output_score['final_score']['scores'], args.binning_threshold)
        
    if args.train=='True':
        output_score['final_score']['correlation'] = correlation_result(output_score['final_score']['scores'], gt_score)
        print(output_score['final_score']['correlation'])
    else:
        pd.DataFrame(data={'pred_score':output_score['final_score']['scores']}).to_csv(os.path.join(default_dir,'summarization.csv') + ".scores", header=False, index=False)
        
    with open(os.path.join(default_dir,'summarization.json'), 'w') as file:
        json.dump(output_score, file, indent=4)

    with open(os.path.join(default_dir,'summarization.json.description'), 'w') as file:
        
        description = make_description(args.template, args.definition, aspect_names, 
                         args.model_name, args.score_func, args.score_temperature,
                         args.initial
                         )
        
        json.dump(description, file, indent=4)

                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # 경로 관련(load)
    parser.add_argument('--model_name', type=str, default='OpenOrca-Platypus2-13B')
    parser.add_argument('--data_path', type=str, default='../data/summarization/train_summarization.tsv')
    parser.add_argument('--template_path', type=str, default='prompt_template.json')
    
    # 경로 관련(save)
    parser.add_argument('--prompt_output_path', type=str, default='prompt_output.json')
    parser.add_argument('--score_output_path', type=str, default='score_output.json')
    parser.add_argument('--submit_result_path', type=str, default='sumbmit_result')
    parser.add_argument('--exp_name', type=str, default='summeval_def_summeval_temp')

    # sampling 관련
    parser.add_argument('--sampling', type=bool, default=False, 
                        help='True: n번 생성, False: n개 생성')
    
    # scoring 관련
    parser.add_argument('--score_func', type=str, default='logprob_sum', 
                        help='direct_generation: 단일 생성 score 사용/ logprob_sum: 1-5의 생성 확률로 가중합/ sampling_sum: n개 생성한 후 평균')
    parser.add_argument('--score_temperature', type=float, default=0.0)
    parser.add_argument('--score_num', type=int, default=20, 
                        help='score_func이 sampling_sum일 때 생성할 개수')
    parser.add_argument('--score_logprobs', type=int, default=None, 
                        help='score_func이 logprob_sum일 때 몇 순위의 logprob까지 확인할지')
    
    parser.add_argument('--train', type=str, default='True')
    parser.add_argument('--definition', type=str, default='summeval')
    parser.add_argument('--template', type=str, default='summeval')
    parser.add_argument('--aspect_list', type=str, default='consistency fluency relevance coherence', nargs='+')
    parser.add_argument('--module', type=str, default='vllm', choices=['vllm', 'guidance'])
    parser.add_argument('--device', type=str, default='1')
    parser.add_argument('--initial', type=str, default='SH')

    parser.add_argument('--rm_redundancy', type=str, default='True')
    parser.add_argument('--start', type=str, default='0')
    parser.add_argument('--end', type=str, default='100')
    parser.add_argument('--max_tok', type=int, default=5)
    
    parser.add_argument('--example', type=str, default='False')
    parser.add_argument('--example_score', type=int, default=1)
    parser.add_argument('--explanation', type=str, default='False')
    
    parser.add_argument('--binning', type=bool, default=False)
    parser.add_argument('--binning_threshold', type=int, default=10)

    args = parser.parse_args()

    print(args.aspect_list)

    os.makedirs(args.score_output_path, exist_ok=True)
    
    main(args)