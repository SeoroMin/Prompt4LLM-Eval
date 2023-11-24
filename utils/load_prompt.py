import sys
from model_dict import *
import json

from collections import defaultdict
from eval4nlp.sangmin.score_func import *

# load aspect definition
def load_definition(template_path,
                     aspect='coherence',
                     definition_type='summeval'):
    
    with open(template_path, 'r') as file:
        templates = json.load(file)
        
    definition = templates['aspect'][aspect][definition_type]
    return definition

# custom prompt
def make_prompt_type(templates, u_prompt, a_prompt, aspect, definition, source, summary, 
                     prompting, n_aspect, scoring, aspect_cate, task_desc_type):
    
    templates=templates
    task_description=templates["task_description"][task_desc_type]
    
    if n_aspect == 'single_aspect':
        template = templates['experiment']['zero_shot'][prompting][n_aspect][scoring][aspect_cate]
        result = template.format(u_prompt=u_prompt,
                        a_prompt=a_prompt,
                        source=source,
                        summary=summary)
        return result
    elif n_aspect == 'multi_aspect':
        template = templates['experiment']['zero_shot'][prompting][n_aspect][f'{scoring}_definition']
        result = template.format(u_prompt=u_prompt,
                        a_prompt=a_prompt,
                        aspect=aspect,
                        definition=definition,
                        source=source,
                        summary=summary,
                        task_description=task_description)
        return result

