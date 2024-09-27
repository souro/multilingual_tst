#!/bin/bash

# start_time=$(date +%s)
# ############################################################
python mtst.py --task pos_to_neg --lang en --src_lang_code en_XX --trg_lang_code en_XX --methodology parallel --src POSITIVE --trg NEGATIVE --test_src POSITIVE --test_trg NEGATIVE --test_print_src POSITIVE --output_file ../output/pos_to_neg-en-parallel.csv
python mtst.py --task neg_to_pos --lang en --src_lang_code en_XX --trg_lang_code en_XX --methodology parallel --src NEGATIVE --trg POSITIVE --test_src NEGATIVE --test_trg POSITIVE --test_print_src NEGATIVE --output_file ../output/neg_to_pos-en-parallel.csv

python mtst.py --task pos_to_neg --lang en --src_lang_code en_XX --trg_lang_code en_XX --methodology ae --src NEGATIVE --trg NEGATIVE --test_src POSITIVE --test_trg NEGATIVE --test_print_src POSITIVE --output_file ../output/pos_to_neg-en-ae.csv
python mtst.py --task neg_to_pos --lang en --src_lang_code en_XX --trg_lang_code en_XX --methodology ae --src POSITIVE --trg POSITIVE --test_src NEGATIVE --test_trg POSITIVE --test_print_src NEGATIVE --output_file ../output/neg_to_pos-en-ae.csv

python mtst.py --task pos_to_neg --lang en --src_lang_code hi_IN --trg_lang_code en_XX --methodology bt --src NEGATIVE_TR --trg NEGATIVE_TR_TR --test_src POSITIVE_TR --test_trg NEGATIVE --test_print_src POSITIVE --output_file ../output/pos_to_neg-en-bt.csv
python mtst.py --task neg_to_pos --lang en --src_lang_code hi_IN --trg_lang_code en_XX --methodology bt --src POSITIVE_TR --trg POSITIVE_TR_TR --test_src NEGATIVE_TR --test_trg POSITIVE --test_print_src NEGATIVE --output_file ../output/neg_to_pos-en-bt.csv

python mtst.py --task pos_to_neg --lang en --src_lang_code en_XX --trg_lang_code en_XX --methodology ae_mask --src NEGATIVE_MASK --trg NEGATIVE --test_src POSITIVE_MASK --test_trg NEGATIVE --test_print_src POSITIVE --output_file ../output/pos_to_neg-en-ae_mask.csv
python mtst.py --task neg_to_pos --lang en --src_lang_code en_XX --trg_lang_code en_XX --methodology ae_mask --src POSITIVE_MASK --trg POSITIVE --test_src NEGATIVE_MASK --test_trg POSITIVE --test_print_src NEGATIVE --output_file ../output/neg_to_pos-en-ae_mask.csv

python mtst.py --task pos_to_neg --lang en --src_lang_code hi_IN --trg_lang_code en_XX --methodology bt_mask --src NEGATIVE_TR_MASK --trg NEGATIVE_TR_TR --test_src POSITIVE_TR_MASK --test_trg NEGATIVE --test_print_src POSITIVE --output_file ../output/pos_to_neg-en-bt_mask.csv
python mtst.py --task neg_to_pos --lang en --src_lang_code hi_IN --trg_lang_code en_XX --methodology bt_mask --src POSITIVE_TR_MASK --trg POSITIVE_TR_TR --test_src NEGATIVE_TR_MASK --test_trg POSITIVE --test_print_src NEGATIVE --output_file ../output/neg_to_pos-en-bt_mask.csv
# ######################################################################################
# # end_time_en=$(date +%s)
# # execution_time=$((end_time_en - start_time))

# # hours=$((execution_time / 3600))
# # minutes=$(( (execution_time % 3600) / 60 ))
# # seconds=$((execution_time % 60))

# # echo "Execution time:" $hours $minutes $seconds

# ######################################################################################
python mtst.py --task pos_to_neg --lang hi --src_lang_code hi_IN --trg_lang_code hi_IN --methodology parallel --src POSITIVE --trg NEGATIVE --test_src POSITIVE --test_trg NEGATIVE --test_print_src POSITIVE --output_file ../output/pos_to_neg-hi-parallel.csv
python mtst.py --task neg_to_pos --lang hi --src_lang_code hi_IN --trg_lang_code hi_IN --methodology parallel --src NEGATIVE --trg POSITIVE --test_src NEGATIVE --test_trg POSITIVE --test_print_src NEGATIVE --output_file ../output/neg_to_pos-hi-parallel.csv

python mtst.py --task pos_to_neg --lang hi --src_lang_code hi_IN --trg_lang_code hi_IN --methodology ae --src NEGATIVE --trg NEGATIVE --test_src POSITIVE --test_trg NEGATIVE --test_print_src POSITIVE --output_file ../output/pos_to_neg-hi-ae.csv
python mtst.py --task neg_to_pos --lang hi --src_lang_code hi_IN --trg_lang_code hi_IN --methodology ae --src POSITIVE --trg POSITIVE --test_src NEGATIVE --test_trg POSITIVE --test_print_src NEGATIVE --output_file ../output/neg_to_pos-hi-ae.csv

python mtst.py --task pos_to_neg --lang hi --src_lang_code en_XX --trg_lang_code hi_IN --methodology bt --src NEGATIVE_TR --trg NEGATIVE_TR_TR --test_src POSITIVE_TR --test_trg NEGATIVE --test_print_src POSITIVE --output_file ../output/pos_to_neg-hi-bt.csv
python mtst.py --task neg_to_pos --lang hi --src_lang_code en_XX --trg_lang_code hi_IN --methodology bt --src POSITIVE_TR --trg POSITIVE_TR_TR --test_src NEGATIVE_TR --test_trg POSITIVE --test_print_src NEGATIVE --output_file ../output/neg_to_pos-hi-bt.csv

python mtst.py --task pos_to_neg --lang hi --src_lang_code hi_IN --trg_lang_code hi_IN --methodology ae_mask --src NEGATIVE_MASK --trg NEGATIVE --test_src POSITIVE_MASK --test_trg NEGATIVE --test_print_src POSITIVE --output_file ../output/pos_to_neg-hi-ae_mask.csv
python mtst.py --task neg_to_pos --lang hi --src_lang_code hi_IN --trg_lang_code hi_IN --methodology ae_mask --src POSITIVE_MASK --trg POSITIVE --test_src NEGATIVE_MASK --test_trg POSITIVE --test_print_src NEGATIVE --output_file ../output/neg_to_pos-hi-ae_mask.csv

python mtst.py --task pos_to_neg --lang hi --src_lang_code en_XX --trg_lang_code hi_IN --methodology bt_mask --src NEGATIVE_TR_MASK --trg NEGATIVE_TR_TR --test_src POSITIVE_TR_MASK --test_trg NEGATIVE --test_print_src POSITIVE --output_file ../output/pos_to_neg-hi-bt_mask.csv
python mtst.py --task neg_to_pos --lang hi --src_lang_code en_XX --trg_lang_code hi_IN --methodology bt_mask --src POSITIVE_TR_MASK --trg POSITIVE_TR_TR --test_src NEGATIVE_TR_MASK --test_trg POSITIVE --test_print_src NEGATIVE --output_file ../output/neg_to_pos-hi-bt_mask.csv
######################################################################################
python mtst.py --task pos_to_neg --lang mr --batch_size 16  --src_lang_code mr_IN --trg_lang_code mr_IN --methodology parallel --src POSITIVE --trg NEGATIVE --test_src POSITIVE --test_trg NEGATIVE --test_print_src POSITIVE --output_file ../output/pos_to_neg-mr-parallel.csv
python mtst.py --task neg_to_pos --lang mr --batch_size 16  --src_lang_code mr_IN --trg_lang_code mr_IN --methodology parallel --src NEGATIVE --trg POSITIVE --test_src NEGATIVE --test_trg POSITIVE --test_print_src NEGATIVE --output_file ../output/neg_to_pos-mr-parallel.csv

python mtst.py --task pos_to_neg --lang mr --batch_size 16  --src_lang_code mr_IN --trg_lang_code mr_IN --methodology ae --src NEGATIVE --trg NEGATIVE --test_src POSITIVE --test_trg NEGATIVE --test_print_src POSITIVE --output_file ../output/pos_to_neg-mr-ae.csv
python mtst.py --task neg_to_pos --lang mr --batch_size 16  --src_lang_code mr_IN --trg_lang_code mr_IN --methodology ae --src POSITIVE --trg POSITIVE --test_src NEGATIVE --test_trg POSITIVE --test_print_src NEGATIVE --output_file ../output/neg_to_pos-mr-ae.csv

python mtst.py --task pos_to_neg --lang mr --batch_size 16  --src_lang_code en_XX --trg_lang_code mr_IN --methodology bt --src NEGATIVE_TR --trg NEGATIVE_TR_TR --test_src POSITIVE_TR --test_trg NEGATIVE --test_print_src POSITIVE --output_file ../output/pos_to_neg-mr-bt.csv
python mtst.py --task neg_to_pos --lang mr --batch_size 16  --src_lang_code en_XX --trg_lang_code mr_IN --methodology bt --src POSITIVE_TR --trg POSITIVE_TR_TR --test_src NEGATIVE_TR --test_trg POSITIVE --test_print_src NEGATIVE --output_file ../output/neg_to_pos-mr-bt.csv

python mtst.py --task pos_to_neg --lang mr --batch_size 16  --src_lang_code mr_IN --trg_lang_code mr_IN --methodology ae_mask --src NEGATIVE_MASK --trg NEGATIVE --test_src POSITIVE_MASK --test_trg NEGATIVE --test_print_src POSITIVE --output_file ../output/pos_to_neg-mr-ae_mask.csv
python mtst.py --task neg_to_pos --lang mr --batch_size 16  --src_lang_code mr_IN --trg_lang_code mr_IN --methodology ae_mask --src POSITIVE_MASK --trg POSITIVE --test_src NEGATIVE_MASK --test_trg POSITIVE --test_print_src NEGATIVE --output_file ../output/neg_to_pos-mr-ae_mask.csv

python mtst.py --task pos_to_neg --lang mr --batch_size 16  --src_lang_code en_XX --trg_lang_code mr_IN --methodology bt_mask --src NEGATIVE_TR_MASK --trg NEGATIVE_TR_TR --test_src POSITIVE_TR_MASK --test_trg NEGATIVE --test_print_src POSITIVE --output_file ../output/pos_to_neg-hmr-bt_mask.csv
python mtst.py --task neg_to_pos --lang mr --batch_size 16  --src_lang_code en_XX --trg_lang_code mr_IN --methodology bt_mask --src POSITIVE_TR_MASK --trg POSITIVE_TR_TR --test_src NEGATIVE_TR_MASK --test_trg POSITIVE --test_print_src NEGATIVE --output_file ../output/neg_to_pos-mr-bt_mask.csv
# # ######################################################################################
python mtst.py --task pos_to_neg --lang ml --src_lang_code ml_IN --trg_lang_code ml_IN --methodology parallel --src POSITIVE --trg NEGATIVE --test_src POSITIVE --test_trg NEGATIVE --test_print_src POSITIVE --output_file ../output/pos_to_neg-ml-parallel.csv
python mtst.py --task neg_to_pos --lang ml --src_lang_code ml_IN --trg_lang_code ml_IN --methodology parallel --src NEGATIVE --trg POSITIVE --test_src NEGATIVE --test_trg POSITIVE --test_print_src NEGATIVE --output_file ../output/neg_to_pos-ml-parallel.csv

python mtst.py --task pos_to_neg --lang ml --src_lang_code ml_IN --trg_lang_code ml_IN --methodology ae --src NEGATIVE --trg NEGATIVE --test_src POSITIVE --test_trg NEGATIVE --test_print_src POSITIVE --output_file ../output/pos_to_neg-ml-ae.csv
python mtst.py --task neg_to_pos --lang ml --src_lang_code ml_IN --trg_lang_code ml_IN --methodology ae --src POSITIVE --trg POSITIVE --test_src NEGATIVE --test_trg POSITIVE --test_print_src NEGATIVE --output_file ../output/neg_to_pos-ml-ae.csv

python mtst.py --task pos_to_neg --lang ml --src_lang_code en_XX --trg_lang_code ml_IN --methodology bt --src NEGATIVE_TR --trg NEGATIVE_TR_TR --test_src POSITIVE_TR --test_trg NEGATIVE --test_print_src POSITIVE --output_file ../output/pos_to_neg-hi-ml.csv
python mtst.py --task neg_to_pos --lang ml --src_lang_code en_XX --trg_lang_code ml_IN --methodology bt --src POSITIVE_TR --trg POSITIVE_TR_TR --test_src NEGATIVE_TR --test_trg POSITIVE --test_print_src NEGATIVE --output_file ../output/neg_to_pos-ml-bt.csv

python mtst.py --task pos_to_neg --lang ml --src_lang_code ml_IN --trg_lang_code ml_IN --methodology ae_mask --src NEGATIVE_MASK --trg NEGATIVE --test_src POSITIVE_MASK --test_trg NEGATIVE --test_print_src POSITIVE --output_file ../output/pos_to_neg-ml-ae_mask.csv
python mtst.py --task neg_to_pos --lang ml --src_lang_code ml_IN --trg_lang_code ml_IN --methodology ae_mask --src POSITIVE_MASK --trg POSITIVE --test_src NEGATIVE_MASK --test_trg POSITIVE --test_print_src NEGATIVE --output_file ../output/neg_to_pos-ml-ae_mask.csv

python mtst.py --task pos_to_neg --lang ml --src_lang_code en_XX --trg_lang_code ml_IN --methodology bt_mask --src NEGATIVE_TR_MASK --trg NEGATIVE_TR_TR --test_src POSITIVE_TR_MASK --test_trg NEGATIVE --test_print_src POSITIVE --output_file ../output/pos_to_neg-ml-bt_mask.csv
python mtst.py --task neg_to_pos --lang ml --src_lang_code en_XX --trg_lang_code ml_IN --methodology bt_mask --src POSITIVE_TR_MASK --trg POSITIVE_TR_TR --test_src NEGATIVE_TR_MASK --test_trg POSITIVE --test_print_src NEGATIVE --output_file ../output/neg_to_pos-ml-bt_mask.csv
# # ######################################################################################
python mtst.py --task pos_to_neg --lang ur --src_lang_code ur_PK --trg_lang_code ur_PK --methodology parallel --src POSITIVE --trg NEGATIVE --test_src POSITIVE --test_trg NEGATIVE --test_print_src POSITIVE --output_file ../output/pos_to_neg-ur-parallel.csv
python mtst.py --task neg_to_pos --lang ur --src_lang_code ur_PK --trg_lang_code ur_PK --methodology parallel --src NEGATIVE --trg POSITIVE --test_src NEGATIVE --test_trg POSITIVE --test_print_src NEGATIVE --output_file ../output/neg_to_pos-ur-parallel.csv

python mtst.py --task pos_to_neg --lang ur --src_lang_code ur_PK --trg_lang_code ur_PK --methodology ae --src NEGATIVE --trg NEGATIVE --test_src POSITIVE --test_trg NEGATIVE --test_print_src POSITIVE --output_file ../output/pos_to_neg-ur-ae.csv
python mtst.py --task neg_to_pos --lang ur --src_lang_code ur_PK --trg_lang_code ur_PK --methodology ae --src POSITIVE --trg POSITIVE --test_src NEGATIVE --test_trg POSITIVE --test_print_src NEGATIVE --output_file ../output/neg_to_pos-ur-ae.csv

python mtst.py --task pos_to_neg --lang ur --src_lang_code en_XX --trg_lang_code ur_PK --methodology bt --src NEGATIVE_TR --trg NEGATIVE_TR_TR --test_src POSITIVE_TR --test_trg NEGATIVE --test_print_src POSITIVE --output_file ../output/pos_to_neg-hi-bt.csv
python mtst.py --task neg_to_pos --lang ur --src_lang_code en_XX --trg_lang_code ur_PK --methodology bt --src POSITIVE_TR --trg POSITIVE_TR_TR --test_src NEGATIVE_TR --test_trg POSITIVE --test_print_src NEGATIVE --output_file ../output/neg_to_pos-ur-bt.csv

python mtst.py --task pos_to_neg --lang ur --src_lang_code ur_PK --trg_lang_code ur_PK --methodology ae_mask --src NEGATIVE_MASK --trg NEGATIVE --test_src POSITIVE_MASK --test_trg NEGATIVE --test_print_src POSITIVE --output_file ../output/pos_to_neg-ur-ae_mask.csv
python mtst.py --task neg_to_pos --lang ur --src_lang_code ur_PK --trg_lang_code ur_PK --methodology ae_mask --src POSITIVE_MASK --trg POSITIVE --test_src NEGATIVE_MASK --test_trg POSITIVE --test_print_src NEGATIVE --output_file ../output/neg_to_pos-ur-ae_mask.csv

python mtst.py --task pos_to_neg --lang ur --src_lang_code en_XX --trg_lang_code ur_PK --methodology bt_mask --src NEGATIVE_TR_MASK --trg NEGATIVE_TR_TR --test_src POSITIVE_TR_MASK --test_trg NEGATIVE --test_print_src POSITIVE --output_file ../output/pos_to_neg-ur-bt_mask-parallel.csv
python mtst.py --task neg_to_pos --lang ur --src_lang_code en_XX --trg_lang_code ur_PK --methodology bt_mask --src POSITIVE_TR_MASK --trg POSITIVE_TR_TR --test_src NEGATIVE_TR_MASK --test_trg POSITIVE --test_print_src NEGATIVE --output_file ../output/neg_to_pos-ur-bt_mask.csv
# # ######################################################################################
python mtst.py --task pos_to_neg --lang mag --src_lang_code hi_IN --trg_lang_code hi_IN --methodology parallel --src POSITIVE --trg NEGATIVE --test_src POSITIVE --test_trg NEGATIVE --test_print_src POSITIVE --output_file ../output/pos_to_neg-mag-parallel.csv
python mtst.py --task neg_to_pos --lang mag --src_lang_code hi_IN --trg_lang_code hi_IN --methodology parallel --src NEGATIVE --trg POSITIVE --test_src NEGATIVE --test_trg POSITIVE --test_print_src NEGATIVE --output_file ../output/neg_to_pos-mag-parallel.csv

python mtst.py --task pos_to_neg --lang mag --src_lang_code hi_IN --trg_lang_code hi_IN --methodology ae --src NEGATIVE --trg NEGATIVE --test_src POSITIVE --test_trg NEGATIVE --test_print_src POSITIVE --output_file ../output/pos_to_neg-mag-ae.csv
python mtst.py --task neg_to_pos --lang mag --src_lang_code hi_IN --trg_lang_code hi_IN --methodology ae --src POSITIVE --trg POSITIVE --test_src NEGATIVE --test_trg POSITIVE --test_print_src NEGATIVE --output_file ../output/neg_to_pos-mag-ae.csv

python mtst.py --task pos_to_neg --lang mag --src_lang_code en_XX --trg_lang_code hi_IN --methodology bt --src NEGATIVE_TR --trg NEGATIVE_TR_TR --test_src POSITIVE_TR --test_trg NEGATIVE --test_print_src POSITIVE --output_file ../output/pos_to_neg-mag-bt.csv
python mtst.py --task neg_to_pos --lang mag --src_lang_code en_XX --trg_lang_code hi_IN --methodology bt --src POSITIVE_TR --trg POSITIVE_TR_TR --test_src NEGATIVE_TR --test_trg POSITIVE --test_print_src NEGATIVE --output_file ../output/neg_to_pos-mag-bt.csv

python mtst.py --task pos_to_neg --lang mag --src_lang_code hi_IN --trg_lang_code hi_IN --methodology ae_mask --src NEGATIVE_MASK --trg NEGATIVE --test_src POSITIVE_MASK --test_trg NEGATIVE --test_print_src POSITIVE --output_file ../output/pos_to_neg-mag-ae_mask.csv
python mtst.py --task neg_to_pos --lang mag --src_lang_code hi_IN --trg_lang_code hi_IN --methodology ae_mask --src POSITIVE_MASK --trg POSITIVE --test_src NEGATIVE_MASK --test_trg POSITIVE --test_print_src NEGATIVE --output_file ../output/neg_to_pos-mag-ae_mask.csv

python mtst.py --task pos_to_neg --lang mag --src_lang_code en_XX --trg_lang_code hi_IN --methodology bt_mask --src NEGATIVE_TR_MASK --trg NEGATIVE_TR_TR --test_src POSITIVE_TR_MASK --test_trg NEGATIVE --test_print_src POSITIVE --output_file ../output/pos_to_neg-mag-bt_mask.csv
python mtst.py --task neg_to_pos --lang mag --src_lang_code en_XX --trg_lang_code hi_IN --methodology bt_mask --src POSITIVE_TR_MASK --trg POSITIVE_TR_TR --test_src NEGATIVE_TR_MASK --test_trg POSITIVE --test_print_src NEGATIVE --output_file ../output/neg_to_pos-mag-bt_mask.csv
# # ######################################################################################
python mtst.py --task pos_to_neg --lang pa --src_lang_code hi_IN --trg_lang_code hi_IN --methodology parallel --src POSITIVE --trg NEGATIVE --test_src POSITIVE --test_trg NEGATIVE --test_print_src POSITIVE --output_file ../output/pos_to_neg-pa-parallel.csv
python mtst.py --task neg_to_pos --lang pa --src_lang_code hi_IN --trg_lang_code hi_IN --methodology parallel --src NEGATIVE --trg POSITIVE --test_src NEGATIVE --test_trg POSITIVE --test_print_src NEGATIVE --output_file ../output/neg_to_pos-pa-parallel.csv

python mtst.py --task pos_to_neg --lang pa --src_lang_code hi_IN --trg_lang_code hi_IN --methodology ae --src NEGATIVE --trg NEGATIVE --test_src POSITIVE --test_trg NEGATIVE --test_print_src POSITIVE --output_file ../output/pos_to_neg-pa-ae.csv
python mtst.py --task neg_to_pos --lang pa --src_lang_code hi_IN --trg_lang_code hi_IN --methodology ae --src POSITIVE --trg POSITIVE --test_src NEGATIVE --test_trg POSITIVE --test_print_src NEGATIVE --output_file ../output/neg_to_pos-pa-ae.csv

python mtst.py --task pos_to_neg --lang pa --src_lang_code en_XX --trg_lang_code hi_IN --methodology bt --src NEGATIVE_TR --trg NEGATIVE_TR_TR --test_src POSITIVE_TR --test_trg NEGATIVE --test_print_src POSITIVE --output_file ../output/pos_to_neg-pa-bt.csv
python mtst.py --task neg_to_pos --lang pa --src_lang_code en_XX --trg_lang_code hi_IN --methodology bt --src POSITIVE_TR --trg POSITIVE_TR_TR --test_src NEGATIVE_TR --test_trg POSITIVE --test_print_src NEGATIVE --output_file ../output/neg_to_pos-pa-bt.csv

python mtst.py --task pos_to_neg --lang pa --src_lang_code hi_IN --trg_lang_code hi_IN --methodology ae_mask --src NEGATIVE_MASK --trg NEGATIVE --test_src POSITIVE_MASK --test_trg NEGATIVE --test_print_src POSITIVE --output_file ../output/pos_to_neg-pa-ae_mask.csv
python mtst.py --task neg_to_pos --lang pa --src_lang_code hi_IN --trg_lang_code hi_IN --methodology ae_mask --src POSITIVE_MASK --trg POSITIVE --test_src NEGATIVE_MASK --test_trg POSITIVE --test_print_src NEGATIVE --output_file ../output/neg_to_pos-pa-ae_mask.csv

python mtst.py --task pos_to_neg --lang pa --src_lang_code en_XX --trg_lang_code hi_IN --methodology bt_mask --src NEGATIVE_TR_MASK --trg NEGATIVE_TR_TR --test_src POSITIVE_TR_MASK --test_trg NEGATIVE --test_print_src POSITIVE --output_file ../output/pos_to_neg-pa-bt_mask.csv
python mtst.py --task neg_to_pos --lang pa --src_lang_code en_XX --trg_lang_code hi_IN --methodology bt_mask --src POSITIVE_TR_MASK --trg POSITIVE_TR_TR --test_src NEGATIVE_TR_MASK --test_trg POSITIVE --test_print_src NEGATIVE --output_file ../output/neg_to_pos-pa-bt_mask.csv
# # ######################################################################################
python mtst.py --task pos_to_neg --lang or --src_lang_code hi_IN --trg_lang_code hi_IN --methodology parallel --src POSITIVE --trg NEGATIVE --test_src POSITIVE --test_trg NEGATIVE --test_print_src POSITIVE --output_file ../output/pos_to_neg-or-parallel.csv
python mtst.py --task neg_to_pos --lang or --src_lang_code hi_IN --trg_lang_code hi_IN --methodology parallel --src NEGATIVE --trg POSITIVE --test_src NEGATIVE --test_trg POSITIVE --test_print_src NEGATIVE --output_file ../output/neg_to_pos-or-parallel.csv

python mtst.py --task pos_to_neg --lang or --src_lang_code hi_IN --trg_lang_code hi_IN --methodology ae --src NEGATIVE --trg NEGATIVE --test_src POSITIVE --test_trg NEGATIVE --test_print_src POSITIVE --output_file ../output/pos_to_neg-or-ae.csv
python mtst.py --task neg_to_pos --lang or --src_lang_code hi_IN --trg_lang_code hi_IN --methodology ae --src POSITIVE --trg POSITIVE --test_src NEGATIVE --test_trg POSITIVE --test_print_src NEGATIVE --output_file ../output/neg_to_pos-or-ae.csv

python mtst.py --task pos_to_neg --lang or --src_lang_code en_XX --trg_lang_code hi_IN --methodology bt --src NEGATIVE_TR --trg NEGATIVE_TR_TR --test_src POSITIVE_TR --test_trg NEGATIVE --test_print_src POSITIVE --output_file ../output/pos_to_neg-or-bt.csv
python mtst.py --task neg_to_pos --lang or --src_lang_code en_XX --trg_lang_code hi_IN --methodology bt --src POSITIVE_TR --trg POSITIVE_TR_TR --test_src NEGATIVE_TR --test_trg POSITIVE --test_print_src NEGATIVE --output_file ../output/neg_to_pos-or-bt.csv

python mtst.py --task pos_to_neg --lang or --src_lang_code hi_IN --trg_lang_code hi_IN --methodology ae_mask --src NEGATIVE_MASK --trg NEGATIVE --test_src POSITIVE_MASK --test_trg NEGATIVE --test_print_src POSITIVE --output_file ../output/pos_to_neg-or-ae_mask.csv
python mtst.py --task neg_to_pos --lang or --src_lang_code hi_IN --trg_lang_code hi_IN --methodology ae_mask --src POSITIVE_MASK --trg POSITIVE --test_src NEGATIVE_MASK --test_trg POSITIVE --test_print_src NEGATIVE --output_file ../output/neg_to_pos-or-ae_mask.csv

python mtst.py --task pos_to_neg --lang or --src_lang_code en_XX --trg_lang_code hi_IN --methodology bt_mask --src NEGATIVE_TR_MASK --trg NEGATIVE_TR_TR --test_src POSITIVE_TR_MASK --test_trg NEGATIVE --test_print_src POSITIVE --output_file ../output/pos_to_neg-or-bt_mask.csv
python mtst.py --task neg_to_pos --lang or --src_lang_code en_XX --trg_lang_code hi_IN --methodology bt_mask --src POSITIVE_TR_MASK --trg POSITIVE_TR_TR --test_src NEGATIVE_TR_MASK --test_trg POSITIVE --test_print_src NEGATIVE --output_file ../output/neg_to_pos-or-bt_mask.csv
######################################################################################################################
python mtst.py --task pos_to_neg --lang te --src_lang_code te_IN --trg_lang_code te_IN --methodology parallel --src POSITIVE --trg NEGATIVE --test_src POSITIVE --test_trg NEGATIVE --test_print_src POSITIVE --output_file ../output/pos_to_neg-te-parallel.csv
python mtst.py --task neg_to_pos --lang te --src_lang_code te_IN --trg_lang_code te_IN --methodology parallel --src NEGATIVE --trg POSITIVE --test_src NEGATIVE --test_trg POSITIVE --test_print_src NEGATIVE --output_file ../output/neg_to_pos-te-parallel.csv

python mtst.py --task pos_to_neg --lang te --src_lang_code te_IN --trg_lang_code te_IN --methodology ae --src NEGATIVE --trg NEGATIVE --test_src POSITIVE --test_trg NEGATIVE --test_print_src POSITIVE --output_file ../output/pos_to_neg-te-ae.csv
python mtst.py --task neg_to_pos --lang te --src_lang_code te_IN --trg_lang_code te_IN --methodology ae --src POSITIVE --trg POSITIVE --test_src NEGATIVE --test_trg POSITIVE --test_print_src NEGATIVE --output_file ../output/neg_to_pos-te-ae.csv

python mtst.py --task pos_to_neg --lang te --src_lang_code en_XX --trg_lang_code te_IN --methodology bt --src NEGATIVE_TR --trg NEGATIVE_TR_TR --test_src POSITIVE_TR --test_trg NEGATIVE --test_print_src POSITIVE --output_file ../output/pos_to_neg-te-bt.csv
python mtst.py --task neg_to_pos --lang te --src_lang_code en_XX --trg_lang_code te_IN --methodology bt --src POSITIVE_TR --trg POSITIVE_TR_TR --test_src NEGATIVE_TR --test_trg POSITIVE --test_print_src NEGATIVE --output_file ../output/neg_to_pos-te-bt.csv

python mtst.py --task pos_to_neg --lang te --src_lang_code te_IN --trg_lang_code te_IN --methodology ae_mask --src NEGATIVE_MASK --trg NEGATIVE --test_src POSITIVE_MASK --test_trg NEGATIVE --test_print_src POSITIVE --output_file ../output/pos_to_neg-te-ae_mask.csv
python mtst.py --task neg_to_pos --lang te --src_lang_code te_IN --trg_lang_code te_IN --methodology ae_mask --src POSITIVE_MASK --trg POSITIVE --test_src NEGATIVE_MASK --test_trg POSITIVE --test_print_src NEGATIVE --output_file ../output/neg_to_pos-te-ae_mask.csv

python mtst.py --task pos_to_neg --lang te --src_lang_code en_XX --trg_lang_code te_IN --methodology bt_mask --src NEGATIVE_TR_MASK --trg NEGATIVE_TR_TR --test_src POSITIVE_TR_MASK --test_trg NEGATIVE --test_print_src POSITIVE --output_file ../output/pos_to_neg-te-bt_mask.csv
python mtst.py --task neg_to_pos --lang te --src_lang_code en_XX --trg_lang_code te_IN --methodology bt_mask --src POSITIVE_TR_MASK --trg POSITIVE_TR_TR --test_src NEGATIVE_TR_MASK --test_trg POSITIVE --test_print_src NEGATIVE --output_file ../output/neg_to_pos-te-bt_mask.csv