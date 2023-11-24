# Prompt4LLM-Eval

This is the Source Code of Paper: Which is better? [Exploring Prompting Strategy For LLM-based Metrics](https://arxiv.org/abs/2311.03754)


# What is Prompt4LLM-Eval?


# Key findings


# Usage
Setting up an experimental environment
```
conda create --name Eval4NLP23 python=3.10
conda activate Eval4NLP23
#conda install pip
pip install -r requirements
```

Run Experiment with vllm
```
./scripts/inference_vllm.sh
```

Run Experiment with Guidance
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
