#!/bin/bash

# python automatic_eval.py --task pos_to_neg --lang en --output_file ../output/pos_to_neg-en-parallel.csv --methodology parallel

# langs=("en" "hi" "pa" "mag" "ml" "mr" "or" "te" "ur")
# tasks=("pos_to_neg" "neg_to_pos")
# methodologies=("Llama2_7b" "Llama2_7b_chat")

# for lang in "${langs[@]}"; do
#   for task in "${tasks[@]}"; do
#     for methodology in "${methodologies[@]}"; do
#       python automatic_eval_tmp.py --task ${task} --lang ${lang} --output_file ../output/${task}-${lang}-${methodology}.csv --methodology ${methodology}
#     done
#   done
# done
