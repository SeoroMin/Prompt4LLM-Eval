import torch
import torch.nn.functional as F
import scipy.stats
import matplotlib.pyplot as plt
import numpy as np
import re

def weighted_sum(scores):
    # Convert scores to tensor
    scores_tensor = torch.tensor(list(scores.values()), dtype=torch.float32)
    
    # Compute softmax probabilities
    probabilities_tensor = F.softmax(scores_tensor, dim=0)
    
    # Compute weighted sum
    keys_tensor = torch.tensor([float(key) for key in scores.keys()], dtype=torch.float32)
    result = torch.dot(keys_tensor, probabilities_tensor)
    
    return result.item()

# vllm
def generate_standard(model, prompt_list, sampling_params):
    outputs = model.generate(prompt_list, sampling_params)

    # Print the outputs.
    output_list = []
    i = 0
    for output in outputs:
        generated_text = output.outputs[0].text
        output_list.append(generated_text)
        i+=1
        if i<5:
            print(f"Generated text: {generated_text!r}")
    
    return output_list

def generate_sampling(model, prompt, sampling_params, n):
    output_list = []
    i = 0
    for j in range(n):
        outputs = model.generate(prompt, sampling_params)
        generated_text = outputs[0].outputs[0].text
        output_list.append(generated_text)
        i+=1
        if i<3:
            print(f"Generated text: {generated_text!r}")
    
    return output_list

def generate_score(model, prompt_list, sampling_params, score_func, explanation=True):
    outputs = model.generate(prompt_list, sampling_params)
    tokenizer = model.get_tokenizer()

    # Print the outputs.
    output_list = []
    i = 0
    for output in outputs:
        generated_text = output.outputs[0].text
        output_list.append(generated_text)
        i+=1
        if i<5:
            print(f"Generated text: {generated_text!r}")
    
    if score_func=='direct_generation':
        # score_list = [parse_output(x) for x in output_list]
        parse = [parse_output(x) for x in output_list]
        extract = [extract_score(x) for x in output_list]
        parse_extract = parse.copy()
        for i in range(len(parse_extract)):
            if parse_extract[i] == 0:
                parse_extract[i] = extract[i]
        score_list = parse_extract
        
        if explanation:
            explain_list = [extract_rationale(x) for x in output_list]
            
            return score_list, explain_list
        
        else:
            return score_list
        
    elif score_func=='logprob_sum':
        score_list = []
        
        score_token_id = {'1':tokenizer.convert_tokens_to_ids('1'),
                    '2':tokenizer.convert_tokens_to_ids('2'),
                    '3':tokenizer.convert_tokens_to_ids('3'),
                    '4':tokenizer.convert_tokens_to_ids('4'),
                    '5':tokenizer.convert_tokens_to_ids('5')}
        
        score_prob_dict = {}
        score_prob_dict['1'] = {}
        score_prob_dict['2'] = {}
        score_prob_dict['3'] = {}
        score_prob_dict['4'] = {}
        score_prob_dict['5'] = {}
        
        for output in outputs:
            for n in range(5):
                # 첫번째 토큰 logprob 할당
                log_prob_dict = output.outputs[0].logprobs[0]
                # 토큰 생성확률 중 제일 높은 것이 score_token_id에 포함되어 있으면 해당 위치의 logprob 할당, 그렇지 않으면 첫번째 토큰의 logprob gkfekd
                if list(output.outputs[0].logprobs[n].keys())[0] in list(score_token_id.values()):
                    log_prob_dict = output.outputs[0].logprobs[n]
                    break
            
            for score_token in list(score_token_id.keys()):
                try:
                    prob = log_prob_dict[score_token_id[score_token]]
                except:
                    prob = -10000
                score_prob_dict[score_token] = prob
                
            ws_score = weighted_sum(score_prob_dict)
            score_list.append(ws_score)
        return score_list
                
    elif score_func=='sampling_sum':
        score_list = []
        for output in outputs:
            sampling_list = []
            for output_text in output.outputs:
                sampling_list.append(output_text.text)
            # sampling_parse_list = [parse_output(x) for x in sampling_list]
            
            parse = [parse_output(x) for x in sampling_list]
            extract = [extract_score(x) for x in sampling_list]
            parse_extract = parse.copy()
            for i in range(len(parse_extract)):
                if parse_extract[i] == 0:
                    parse_extract[i] = extract[i]
                    
            sampling_parse_list = parse_extract
            
            score_list.append(np.mean(np.array(sampling_parse_list)))
            
        return score_list





# guidance
def scoring(res, num_outputs, calculation_method, aspects):
    if num_outputs=="single":
        if calculation_method=="dg":
            score = res['score']
        elif calculation_method=="ws":
            score = weighted_sum(res['logprobs'])
    
    elif num_outputs=="multiple":
        if calculation_method=="dg":
            variables = {}
            for aspect in aspects:
                variables[aspect] = int(res[aspect])
            score = str(sum(variables.values()) / len(variables))
        elif calculation_method=="ws":
            variables = {}
            for aspect in aspects:
                variables[aspect] = weighted_sum(res[aspect+'_logprobs'])
            score = str(sum(variables.values()) / len(variables))
            
    return score

def correlation_result(res_scores, ref_scores):
    kendall = scipy.stats.kendalltau(res_scores, ref_scores)[0]
    spearman = scipy.stats.spearmanr(res_scores, ref_scores)[0]
    pearson = scipy.stats.pearsonr(res_scores, ref_scores)[0]

    print('pearson: ', round(pearson, 4))
    print('spearman: ', round(spearman, 4))
    print('kendall: ', round(kendall, 4))
    
    output_dict = {}
    output_dict['pearson'] = round(pearson, 4)
    output_dict['spearman'] = round(spearman, 4)
    output_dict['kendall'] = round(kendall, 4)
    
    return output_dict 

    
    

def plot_histogram(vars_list, colors_list, labels_list, bins=20, figsize=(5, 3), title='Distribution', xlabel='Value', ylabel='Count'):
    """
    Plots histograms for each variable in vars_list.

    Parameters:
    - vars_list: List of lists containing variables
    - colors_list: List of colors for each variable
    - labels_list: List of labels for each variable
    - bins: Number of bins for the histogram
    - figsize: Size of the figure
    - title: Title of the plot
    - xlabel: Label for the x-axis
    - ylabel: Label for the y-axis
    """
    plt.figure(figsize=figsize)

    for var, color, label in zip(vars_list, colors_list, labels_list):
        plt.hist(var, bins=bins, color=color, edgecolor='black', alpha=0.7, label=label)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', linewidth=0.7, alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    


def find_outliers(df, col1, col2, method='difference', threshold_factor=2):
    """
    Find outliers based on the difference or ratio of two columns in a DataFrame.

    Parameters:
    - df: DataFrame containing the data
    - col1: Name of the first column
    - col2: Name of the second column
    - method: Method to compute the difference ('difference' or 'ratio')
    - threshold_factor: Factor to multiply with the standard deviation to determine the threshold

    Returns:
    - mean_value: Mean of the differences or ratios
    - variance_value: Variance of the differences or ratios
    - outlier_indices: List of indices of the outliers
    """
    
    if method == 'difference':
        diff = df[col1] - df[col2]
    elif method == 'ratio':
        diff = df[col1] / df[col2]
    else:
        raise ValueError("Invalid method. Use 'difference' or 'ratio'.")
    
    mean_value = np.mean(diff)
    variance_value = np.var(diff)
    
    threshold = threshold_factor * np.sqrt(variance_value)
    outlier_indices = df.index[np.abs(diff - mean_value) > threshold].tolist()
    
    return mean_value, variance_value, outlier_indices

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

def calculate_zero_ratio(lst):
    zero_count = lst.count(0)
    list_length = len(lst)
    zero_ratio = zero_count / list_length
    return round(zero_ratio, 4)

def extract_score(text):
    # Use regex to find the number immediately following "Score:"
    match = re.search(r'Score:\s*(\d+(?:\.\d+)?)', text)
    if match:
        return float(match.group(1)) if '.' in match.group(1) else int(match.group(1))
    else:
        return 0
    
    
def extract_rationale(text):
    start_index = text.find('Rationale:')
    return text[start_index:]

def binning(input_series, threshold):
    # Count the number of each score
    series = copy.deepcopy(input_series)
    value_counts = series.value_counts().sort_values()

    # Handle if the count is below the threshold
    for score, count in value_counts.items():

        if count < threshold:
            # Find the score closest to the current score
            target_value_counts = dict((k, v) for k, v in value_counts.items() if v >= threshold)
            # If there is an equal score difference, change to the score with the lower coun
            closest_scores = sorted(target_value_counts.items(), key=lambda x: (abs(x[0] - score), x[1])) 
            
            closest_score = closest_scores[1][0]
            series.loc[series == score] = closest_score
            # print(f"{score} -> {closest_score}")
    return series