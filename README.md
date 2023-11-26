# Prompt4LLM-Eval

This is the Source Code of Paper: [Which is better? Exploring Prompting Strategy For LLM-based Metrics](https://arxiv.org/abs/2311.03754).


# What is Prompt4LLM-Eval?
Prompt4LLM-Eval is a methodology that analyzes LLM-based assessment methods by decomposing them into three components: Prompt Strategy, Score Aggregation, Explainability. <br>

We won the summarization track at the [AACL-Eval4NLP workshop](https://eval4nlp.github.io/2023/shared-task.html) with this methodology.

# Key findings
For detailed experimental results, please refer to the paper. <br>
- *Dataset : Summeval dev set*
- *Metric : Kendall's tau correlation to measure similarity to human scores*

**1) Prompt Strategy**<br>
The highest performance is achieved when the prompt is configured similarly to human annotation instructions. <br>
||Orca-7B|Orca-13B|
|---|---|---|
|**Human**|**0.3472**|**0.4468**|
|Model|0.2864|0.3844|
|BaSE|0.2746|0.3891|
- *Human: Prompt consisting of human annotation instructions*
- *Model : The evaluation prompt used for GPT4*
- *Baseline : Task description and Score guide*

**2) Score Aggregation**<br>
The direct generation method achieved the highest performance, while the approximation method exhibited low performance due to noise introduced during sampling. <br>
||Orca-7B|Orca-13B|
|---|---|---|
|**Direct**|**0.3472**|**0.4468**|
|Logprob|0.3296|0.4210|
|Approximation|0.3239|0.4002|
- *Direct: A scoring method that uses the score you create as is*
- *Logprob : weighted sum based on 1~5 token probability*
- *Approximation : Calculate the average after sampling the evaluation score N times*

# Usage
**Setting up an experimental environment**
```
conda create --name Prompt4LLM_Eval python=3.10
conda activate Prompt4LLM_Eval
conda install pip
pip install -r requirements.txt
```

**Running experiment with vllm**
```
./scripts/inference_vllm.sh
```

**Running experiment with Guidance**
```
./scripts/inference_guidance.sh
```

# Bib
```
@misc{kim2023better,
      title={Which is better? Exploring Prompting Strategy For LLM-based Metrics}, 
      author={Joonghoon Kim and Saeran Park and Kiyoon Jeong and Sangmin Lee and Seung Hun Han and Jiyoon Lee and Pilsung Kang},
      year={2023},
      eprint={2311.03754},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
