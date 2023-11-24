model_list="orca_mini_v3_7b OpenOrca-Platypus2-13B Nous-Hermes-13b" # orca_mini_v3_7b OpenOrca-Platypus2-13B Nous-Hermes-13b
definition_list="summeval"
train=True
max_token=5
temperature_list='0'
aspect_category_list="qags" # train test all qags
n_aspect="single_aspect"
scoring_list="five" # five hundred
task_description_list="base_summeval" # role_expert_summeval role_expert2_summeval role_professor_summeval role_linguist_summeval role_child_summeval short_summeval mid_summeval long_summeval


for scoring in ${scoring_list}
do
    for model in ${model_list}
    do
        for definition in ${definition_list}
        do
            for temperature in ${temperature_list}
            do
                for aspect_category in ${aspect_category_list}
                do
                    for task_description in ${task_description_list}
                    do
                        python src/inference_vllm.py \
                        --model_name ${model} \
                        --data_path data_path \
                        --template_path prompt_template.json \
                        --score_func direct_generation \
                        --score_temperature ${temperature} \
                        --score_num 1 \
                        --score_logprobs 0 \
                        --train ${train} \
                        --definition ${definition} \
                        --max_token ${max_token} \
                        --aspect_category ${aspect_category} \
                        --n_aspect ${n_aspect} \
                        --scoring ${scoring} \
                        --task_description ${task_description}
                    done
                done
            done
        done
    done
done