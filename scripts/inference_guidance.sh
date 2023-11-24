score_func_list='sampling_sum' # direct_generation log_probs sampling_sum
aspect_category_list='multi' # qags

for score_func in ${score_func_list}
do
    for aspect_category in ${aspect_category_list}
    do
        python src/guidance_flu.py \
        --score_func ${score_func} \
        --aspect_category ${aspect_category}
    done
done