#!/bin/bash

print_time() {
    echo "$(date +'%H:%M:%S')"
}

convert_seconds() {
    local sec=$1
    local hours=$(( sec / 3600 ))
    local minutes=$(( ( sec % 3600 ) / 60 ))
    local seconds=$(( sec % 60 ))
    printf "%02d:%02d:%02d\n" $hours $minutes $seconds
}

start_time=$(date +%s)

tasks=(pos_to_neg neg_to_pos)
langs=(en hi mag mr or pa te ur ml)
batch_sizes=(1 2 3 4 8 16 32 64)

for task in "${tasks[@]}"; do
    if [ "$task" == "pos_to_neg" ]; then
        src=POSITIVE
        trg=NEGATIVE
        test_src=POSITIVE
        test_trg=NEGATIVE
        test_print_src=POSITIVE
    elif [ "$task" == "neg_to_pos" ]; then
        src=NEGATIVE
        trg=POSITIVE
        test_src=NEGATIVE
        test_trg=POSITIVE
        test_print_src=NEGATIVE
    else
        echo "Invalid task: $task"
        exit 1
    fi
    for lang in "${langs[@]}"; do
        case $lang in
            en)
                src_lang_code=en_XX
                trg_lang_code=en_XX
                ;;
            hi|mag|pa)
                src_lang_code=hi_IN
                trg_lang_code=hi_IN
                ;;
            ml)
                src_lang_code=ml_IN
                trg_lang_code=ml_IN
                ;;
            mr)
                src_lang_code=mr_IN
                trg_lang_code=mr_IN
                ;;
            ur)
                src_lang_code=ur_PK
                trg_lang_code=ur_PK
                ;;
            te)
                src_lang_code=te_IN
                trg_lang_code=te_IN
                ;;
            *)
                src_lang_code=None
                trg_lang_code=None
                ;;
        esac

        for batch_size in "${batch_sizes[@]}"; do
            echo "Script execution started at $(print_time)"

            python mtst.py \
                --task $task \
                --lang $lang \
                --src_lang_code $src_lang_code \
                --trg_lang_code $trg_lang_code \
                --methodology parallel \
                --src $src \
                --trg $trg \
                --test_src $test_src \
                --test_trg $test_trg \
                --test_print_src $test_print_src \
                --output_file ../output_exp/${task}-${lang}-parallel-${batch_size}.csv \
                --batch_size $batch_size

            # Check exit status of Python script
            if [ $? -ne 0 ]; then
                echo "Error: Python script execution failed with arguments: --task $task --lang $lang --batch_size $batch_size"
                exit 1
            fi

            echo "Script execution ended at $(print_time)"

            end_time=$(date +%s)
            iteration_execution_time=$(( end_time - start_time ))

            echo "Execution time for this iteration: $(convert_seconds $iteration_execution_time)"
        done
    done
done

end_time=$(date +%s)
total_execution_time=$(( end_time - start_time ))

echo "Total execution time: $(convert_seconds $total_execution_time)"