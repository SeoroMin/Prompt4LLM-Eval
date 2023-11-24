# import sys
from model_dict import *
from inference import *
# import pandas as pd
# import guidance
import json
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
from vllm import LLM, SamplingParams
# import scipy.stats
from collections import defaultdict
# import re
from utils import *

# 저장되어 있는 aspect definition을 불러오기
def load_definition(template_path,
                     aspect='coherence',
                     definition_type='summeval'):
    
    with open(template_path, 'r') as file:
        templates = json.load(file)
        
    definition = templates['aspect'][aspect][definition_type]
    return definition

# json에 저장되어 있는 prompt format의 key값을 넣고 .format으로 인자 넣어주면 되면
def make_prompt_type(template_path,
                     u_prompt,
                     a_prompt,
                     source,
                     summary,
                     score=True,
                     template_key='simple',
                     task_type='evaluation step',
                     aspect='coherence'):
    
    with open(template_path, 'r') as file:
        templates = json.load(file)
    
    # 저장되어 있는 task_template format 불러오기    
    template = templates.get(template_key)
    
    # 요약문을 포함하여 점수를 매기기 위한 prompt
    if score:
        # scoring을 위한 prompt 만들기
        score_template = template.format(task_type=task_type, 
                               aspect=aspect)
        score_prompt_list = []
        for i in range(len(summary)):
            score_prompt = score_template.format(u_prompt=u_prompt,
                                                 a_prompt=a_prompt,
                                                 source=source[i],
                                                 summary=summary[i])
            score_prompt_list.append(score_prompt)
        return score_prompt_list
    else:
        return template.format(task_type=task_type, 
                               aspect=aspect)



# geval-base prompt
def evaluation_prompt_geval_base(u_prompt, a_prompt, source, summary, 
                                 aspect, definition, evaluation_step, data_type,
                                 start = 0, end = 100
                                 ):
    results = []
    
    if data_type == 'test':
        source_name = 'document'
    else:
        source_name = 'news article'
        

    prompt = f'''{u_prompt}
You will be given one summary written for a {source_name}.

Your task is to evaluate the summary based on a specific metric, rating it on a scale from 1 (worst) to 5 (best).

Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.

Evaluation Criteria:

{definition}

Evaluation Steps:

{evaluation_step}
----
Source text: {source}
Summary: {summary}

{a_prompt}

Score:
'''
    if aspect=='fluency':
        prompt = f'''{u_prompt}
You will be given one summary written for a {source_name}.

Your task is to evaluate the summary based on a specific metric, rating it on a scale from 1 (worst) to 5 (best).

Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.

Evaluation Criteria:

{definition}

----
Source text: {source}
Summary: {summary}

{a_prompt}

Score:
'''
        # results.append(prompt)
    return prompt

def evaluation_prompt_summeval_base(u_prompt, a_prompt, source, summary, 
                                    aspect, definition, data_type,
                                    start = 0, end = 100):
    
    if data_type == 'test':
        source_name = 'document'
    else:
        source_name = 'news article'
        
    prompt = f'''{u_prompt}
In this task you will evaluate the quality of a summary written for a {source_name}.
To correctly solve this task, follow these steps:

1. Carefully read the {source_name}, be aware of the information it contains.
2. Read the proposed summary.
3. Rate each summary on a scale from {str(start)} (worst) to {str(end)} (best) by its {aspect}.

# Definition:
{definition}
----
Source text: {source}
Summary: {summary}

{a_prompt}

Score:
'''
    return prompt


evaluation_prompt_summeval_guidance = '''
{{u_prompt}}
In this task you will evaluate the quality of a summary written for a news article.
To correctly solve this task, follow these steps:

1. Carefully read the news article, be aware of the information it contains.
2. Read the proposed summary.
3. Rate each summary on a scale from 1 (worst) to 5 (best) by its {{aspect}}.

# Definition:
{{definition}}
----
Source text: {{source}}
Summary: {{summary}}

{{a_prompt}}

Score: {{select 'score' options=options logprobs='logprobs'}}
'''


def evaluation_prompt_baseline(u_prompt, a_prompt, source, summary, module):

    if module == 'vllm':
        prompt = f'''{u_prompt}               
    Score the summarization with respect to the summarized document on a continuous scale from 0 to 100, 
    where a score of zero means irrelevant, factually incorrect and not readable and score of one hundred means, relevant, factually correct, good readability

    Source text: {source}
    Summary: {summary}

    {a_prompt}
    Score: 
    '''
    
    else:
        prompt = '''{{u_prompt}}
        Score the summarization with respect to the summarized document on a continuous scale from 0 to 100, 
        where a score of zero means irrelevant, factually incorrect and not readable and score of one hundred means, relevant, factually correct, good readability

        ----
        Source text: {{source}}
        Summary: {{summary}}

        {{a_prompt}}

        Score: {{gen 'score' pattern='(100|[1-9]?[0-9])'}}
        '''   
    
    return prompt

def make_description(template, definition, aspect_list, model, scoring_fn, temperature, initial):
    desc = f'''{template} Template and {definition} Definition. 
    List of aspect used : {aspect_list}
    Model: {model}. Scoring Function: {scoring_fn} Temperature: {temperature}
    {initial}
    '''
    return desc

def remove_redun(prompt_list, model, sampling_params):

    outputs = model.generate(prompt_list, sampling_params)

    yes_no = [x.outputs[0].text.split('\n')[0] for x in outputs]

    explanation_list = []
    for idx, x in enumerate(outputs):

        single_output = x.outputs[0].text.split('\n')

        for idx2, temp in enumerate(single_output):
            if 'Explanation:' in temp:
                explanation = temp
                if temp == 'Explanation:':
                    explanation += ' ' + single_output[idx2+1]

        explanation_list.append(explanation)

    #no_list = [i for i in range(len(yes_no)) if yes_no[i] == 'Yes'] # Redundant samples


    return yes_no, explanation_list

def rm_redun_prompts(u_prompt, summary, a_prompt):
    prompt_list = []

    for sample in summary:

        prompt = f'''{u_prompt}
    In this task you will evaluate the quality of a summary written for a document.

    Provided summary may include direct or rephrased repetitions of the same word or phrase. 

    With that in mind do the following:

    1. Answer whether the summary is redundant or not.
    - Your answer must be in "Yes" or "No" format, where "Yes" means that the summary is redundant and "No" means that the summary is not redundant.

    2. Please provide brief explanation for your answer.
    - Your explanation should only discuss the redundancy of the summary, not the quality of the summary in general.
    ----
    summary: {sample}

    {a_prompt}
        '''
        prompt_list.append(prompt)

    return prompt_list


def rationale_prompt_zero(u_prompt, a_prompt, source, summary, aspect, definition, start = 1, end = 5):
    if aspect=='coherence':
        aspect_verb = 'coherent'
    elif aspect=='consistency':
        aspect_verb = 'consistent'
    elif aspect=='fluency':
        aspect_verb = 'fluent'
    elif aspect=='relevance':
        aspect_verb = 'relevant'
    elif aspect=='factuality':
        aspect_verb = 'factually accurate'
    elif aspect=='readability':
        aspect_verb = 'readable'
    elif aspect=='informativeness':
        aspect_verb = 'informative'
    prompt = f'''{u_prompt}
Your task is to evaluate the {aspect} of a provided summary based on its source document.
Follow these steps:

1. Read the source document
2. Review the summary
3. Analyze for {aspect}
4. Assign a Score: Rate the summary on a scale of {start} to {end}, where:
- {start} means the summary is not {aspect_verb} with the source.
- {end} means the summary is entirely {aspect_verb} with the source.
5. Provide a Rationale: After assigning a score, explain your reasons based on your analysis.

# Definition:
{definition}
----
Source text: {source}
Summary: {summary}
{a_prompt}
'''
    return prompt

def rationale_prompt_few(u_prompt, a_prompt, source, summary, aspect, definition, example_source, example_summary, example_score, example_rationale):
    if aspect=='coherence':
        aspect_verb = 'coherent'
    elif aspect=='consistency':
        aspect_verb = 'consistent'
    elif aspect=='fluency':
        aspect_verb = 'fluent'
    elif aspect=='relevance':
        aspect_verb = 'relevant'
    elif aspect=='factuality':
        aspect_verb = 'factually accurate'
    elif aspect=='readability':
        aspect_verb = 'readable'
    prompt = f'''{u_prompt}
Your task is to evaluate the {aspect} of a provided summary based on its source document.
Follow these steps:

1. Read the source document
2. Review the summary
3. Analyze for {aspect}
4. Assign a Score: Rate the summary on a scale of 1 to 5, where:
- 1 means the summary is not {aspect_verb} with the source.
- 5 means the summary is entirely {aspect_verb} with the source.
5. Provide a Rationale: After assigning a score, explain your reasons based on your analysis.

# Definition:
{definition}

Please refer to following example.
# Example:
Source text: {example_source}
Summary: {example_summary}
{a_prompt}
Score: {example_score}
Rationale: {example_rationale}
----
Source text: {source}
Summary: {summary}
{a_prompt}
'''
    return prompt