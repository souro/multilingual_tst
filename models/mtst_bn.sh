#!/bin/bash

source ../../../env_llm/bin/activate

start_time=$(date +%s)

#######################################################################################
python mtst.py --task pos_to_neg --lang bn --src_lang_code bn_IN --trg_lang_code bn_IN --methodology parallel --src POSITIVE --trg NEGATIVE --test_src POSITIVE --test_trg NEGATIVE --test_print_src POSITIVE --output_file ../output/pos_to_neg-bn-parallel.csv
python mtst.py --task neg_to_pos --lang bn --src_lang_code bn_IN --trg_lang_code bn_IN --methodology parallel --src NEGATIVE --trg POSITIVE --test_src NEGATIVE --test_trg POSITIVE --test_print_src NEGATIVE --output_file ../output/neg_to_pos-bn-parallel.csv

# python mtst.py --task pos_to_neg --lang bn --src_lang_code bn_IN --trg_lang_code bn_IN --methodology ae --src NEGATIVE --trg NEGATIVE --test_src POSITIVE --test_trg NEGATIVE --test_print_src POSITIVE --output_file ../output/pos_to_neg-bn-ae.csv
# python mtst.py --task neg_to_pos --lang bn --src_lang_code bn_IN --trg_lang_code bn_IN --methodology ae --src POSITIVE --trg POSITIVE --test_src NEGATIVE --test_trg POSITIVE --test_print_src NEGATIVE --output_file ../output/neg_to_pos-bn-ae.csv

# python mtst.py --task pos_to_neg --lang bn --src_lang_code en_XX --trg_lang_code bn_IN --methodology bt --src NEGATIVE_TR --trg NEGATIVE_TR_TR --test_src POSITIVE_TR --test_trg NEGATIVE --test_print_src POSITIVE --output_file ../output/pos_to_neg-bn-bt.csv
# python mtst.py --task neg_to_pos --lang bn --src_lang_code en_XX --trg_lang_code bn_IN --methodology bt --src POSITIVE_TR --trg POSITIVE_TR_TR --test_src NEGATIVE_TR --test_trg POSITIVE --test_print_src NEGATIVE --output_file ../output/neg_to_pos-bn-bt.csv

# python mtst.py --task pos_to_neg --lang bn --src_lang_code bn_IN --trg_lang_code bn_IN --methodology ae_mask --src NEGATIVE_MASK --trg NEGATIVE --test_src POSITIVE_MASK --test_trg NEGATIVE --test_print_src POSITIVE --output_file ../output/pos_to_neg-bn-ae_mask.csv
# python mtst.py --task neg_to_pos --lang bn --src_lang_code bn_IN --trg_lang_code bn_IN --methodology ae_mask --src POSITIVE_MASK --trg POSITIVE --test_src NEGATIVE_MASK --test_trg POSITIVE --test_print_src NEGATIVE --output_file ../output/neg_to_pos-bn-ae_mask.csv

# python mtst.py --task pos_to_neg --lang bn --src_lang_code en_XX --trg_lang_code bn_IN --methodology bt_mask --src NEGATIVE_TR_MASK --trg NEGATIVE_TR_TR --test_src POSITIVE_TR_MASK --test_trg NEGATIVE --test_print_src POSITIVE --output_file ../output/pos_to_neg-bn-bt_mask.csv
# python mtst.py --task neg_to_pos --lang bn --src_lang_code en_XX --trg_lang_code bn_IN --methodology bt_mask --src POSITIVE_TR_MASK --trg POSITIVE_TR_TR --test_src NEGATIVE_TR_MASK --test_trg POSITIVE --test_print_src NEGATIVE --output_file ../output/neg_to_pos-bn-bt_mask.csv

######################################################################################

end_time_en=$(date +%s)
execution_time=$((end_time_en - start_time))

hours=$((execution_time / 3600))
minutes=$(( (execution_time % 3600) / 60 ))
seconds=$((execution_time % 60))

echo "Execution time:" $hours $minutes $seconds