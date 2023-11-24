# Prompt4LLM-Eval

This is the Source Code of Paper: [Which is better? Exploring Prompting Strategy For LLM-based Metrics](https://arxiv.org/abs/2311.03754)


# What is Prompt4LLM-Eval?
Prompt4LLM-Eval is a methodology that analyzes LLM-based assessment methods by decomposing them into three components:
- Prompt Strategy
- Score Aggregation
- Explainability. <br>

We won the summarization track at the [AACL-Eval4NLP workshop](https://eval4nlp.github.io/2023/shared-task.html) with this methodology.

# Key findings
For detailed experimental results, see the paper <br>
- *Dataset : Summeval dev set*
- *Metric : Kendall's tau correlation to measure similarity to human scores*

**1) Prompt Strategy**
Highest performance when prompt is configured similarly to a human annotation instruction <br>
||Orca-7B|Orca-13B|
|---|---|---|
|**Human**|**0.3472**|**0.4468**|
|Model|0.2864|0.3844|
|BaSE|0.2746|0.3891|
- *Human: Prompt consisting of human annotation instructions*
- *Model : The evaluation prompt used for GPT4*
- *Baseline : Task description and Score guide*

**2) Score Aggregation**
The direct generation method had the highest performance, and the approximation method had poor performance due to noise during sampling. <br>
||Orca-7B|Orca-13B|
|---|---|---|
|**Direct**|**0.3472**|**0.4468**|
|Logprob|0.3296|0.4210|
|Approximation|0.3239|0.4002|
- *Direct: A scoring method that uses the score you create as is*
- *Logprob : weighted sum based on 1~5 token probability*
- *Approximation : Calculate the average after sampling the evaluation score N times*

**3) Explainability**
When we configured Prompt to ask LLM to generate a rationale, we found that they had the ability to provide a rationale correctly, and hallucinations were reduced when they provided a high quality example. <br>
||Base|Reason-best|
|---|---|---|
|**Good**|**50%**|**69%**|
|Inconsistent|11%|17%|
|**Hallucination**|**36%**|**6%**|
|Different Aspect|6%|8%|
- *Good : Score and rationale match source text and hypothesis text*
- *Complex : Score and rationale are different*
- *Hallucination : When the contents of the source text and hypothesis text are not matched*
- *Different Aspect : Explanation for another aspect other than the one in question*

# Usage
**Setting up an experimental environment**
```
conda create --name Eval4NLP23 python=3.10
conda activate Eval4NLP23
conda install pip
pip install -r requirements.txt
```

**Run Experiment with vllm**
```
./scripts/inference_vllm.sh
```

**Run Experiment with Guidance**
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
