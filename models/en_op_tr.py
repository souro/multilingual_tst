import argparse
# import logger
import time
import torch
import shutil
import random
import socket
import pandas as pd
import numpy as np
from my_logger import logger
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from automatic_eval import TSTEvaluator

# logger.basicConfig(filename='en_op_tr.log', level=logger.INFO)

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def main(args):
    shutil.rmtree('facebook', ignore_errors=True)

    seed_value = 53
    setup_seed(seed_value)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device {device}')    

    if torch.cuda.is_available():
        logger.info(f'GPU machine name: {torch.cuda.get_device_name(0)}')
        logger.info(f'SSH machine name: {socket.gethostname()}')
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    model.to(device)

    df_en = pd.read_csv(args.en_csv)
    df_other_lang = pd.read_csv(args.lang_csv)

    inputs = tokenizer(df_en['pred'].to_list(), return_tensors="pt", padding=True, truncation=True, max_length=30).to(device)
    translated_tokens = model.generate(
        **inputs, forced_bos_token_id=tokenizer.lang_code_to_id[args.lang_code], max_length=30
    )
    translated_texts = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)

    df_output = df_other_lang[['src', 'trg']]
    df_output['pred'] = translated_texts
    df_output.to_csv(args.output_file, index=False)

    tst_evaluator = TSTEvaluator(
        task=args.task,
        lang=args.lang,
        output_file=args.output_file,
        methodology=args.methodology
    )
    
    logger.info('Starting evaluation process.')
    tst_evaluator.set_seed(53)
    accuracy, similarity_score, bleu, fluency_scores_mgpt, fluency_scores_xlmr = tst_evaluator.evaluate()
    
    accuracy_text = f", Sentiment Accuracy: {accuracy:.2f}" if accuracy is not None else ", Sentiment Accuracy: None"
    similarity_score_text = f", Similarity: {similarity_score:.2f}" if similarity_score is not None else ", Similarity: None"
    bleu_text = f", Bleu Score: {bleu:.2f}" if bleu is not None else ", Bleu Score: None"

    fluency_mgpt_text = ", ".join([f"MGPT {key}: {value:.2f}" if value is not None else f"MGPT {key}: None" for key, value in fluency_scores_mgpt.items()])
    fluency_xlmr_text = ", ".join([f"XLM-R {key}: {value:.2f}" if value is not None else f"XLM-R {key}: None" for key, value in fluency_scores_xlmr.items()])
    
    logger.info(f"Results Summary for Language: {args.lang}, Methodology: {args.methodology}, task: {args.task}, Batch Size: {args.batch_size} - {accuracy_text}{similarity_score_text}{bleu_text},{fluency_mgpt_text}, {fluency_xlmr_text}")    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='English Output Translate')

    parser.add_argument('--model_name', type=str, default='facebook/nllb-200-3.3B', help='Pretrained translation model name')
    parser.add_argument('--task', type=str, required=True, help='Task (pos_to_neg or neg_to_pos')
    parser.add_argument('--lang', type=str, required=True, help='Language')
    parser.add_argument('--en_csv', type=str, required=True, help='Path to English CSV file')
    parser.add_argument('--lang_csv', type=str, required=True, help='Path to Language CSV file')
    parser.add_argument('--lang_code', type=str, required=True, help='Language code')
    parser.add_argument('--output_file', type=str, required=True, help='Path to the output CSV file')
    parser.add_argument('--methodology', type=str, default='en-op-tr', help='methodology used for testing')
    parser.add_argument('--batch_size', type=int, default=0, help='batch size for training')

    args = parser.parse_args()

    logger.info("Command-line arguments:")
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")

    start_time = time.time()
    
    logger.info('Starting translation process.')
    
    main(args)
    
    logger.info('Translation and evaluation process completed.')

    end_time = time.time()
    time_taken_seconds = end_time - start_time
    time_taken_formatted = time.strftime('%H:%M:%S', time.gmtime(time_taken_seconds))

    logger.info(f"Time taken for language: {args.lang}, methodology: {args.methodology}, task: {args.task} - {time_taken_formatted}")
    
    logger.info('=' * 50 + '\n')   