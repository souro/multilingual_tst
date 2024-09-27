#!/usr/bin/env python
# coding: utf-8

import shutil
import pandas as pd
import random
import numpy as np
import torch
import argparse
# import logger
import time
import socket
import re
from my_logger import logger
from automatic_eval import TSTEvaluator
from torch.utils.data import DataLoader, TensorDataset
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

# logger.basicConfig(filename='joint.log', level=logger.INFO)

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_data(data_path, langs):
    logger.info("Loading data...")
    train_df_dict = {}
    dev_df_dict = {}
    test_df_dict = {}

    for lang in langs:
        train_df = pd.read_csv(f'{data_path}{lang}_train_025.csv')
        dev_df = pd.read_csv(f'{data_path}{lang}_dev_025.csv')
        test_df = pd.read_csv(f'{data_path}{lang}_test_025.csv')

        train_df['POSITIVE'] = f'<{lang}> ' + train_df['POSITIVE']
        train_df['NEGATIVE'] = f'<{lang}> ' + train_df['NEGATIVE']

        dev_df['POSITIVE'] = f'<{lang}> ' + dev_df['POSITIVE']
        dev_df['NEGATIVE'] = f'<{lang}> ' + dev_df['NEGATIVE']

        test_df['POSITIVE'] = f'<{lang}> ' + test_df['POSITIVE']
        test_df['NEGATIVE'] = f'<{lang}> ' + test_df['NEGATIVE']

        train_df_dict[lang] = train_df
        dev_df_dict[lang] = dev_df
        test_df_dict[lang] = test_df

    train_df = pd.concat(train_df_dict.values(), ignore_index=True)
    dev_df = pd.concat(dev_df_dict.values(), ignore_index=True)

    logger.info("Data loaded successfully.")
    return train_df, dev_df, test_df_dict

class CreateDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels['input_ids'][idx])
            return item

        def __len__(self):
            return len(self.labels['input_ids'])

def gen(src, tokenizer, model):
        src_tknz = tokenizer(src, truncation=True, padding=True, max_length=128, return_tensors='pt')
        generated_ids = model.generate(src_tknz["input_ids"].cuda(), max_length=128)
        return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

def remove_tags(sentence):
    pattern = r'<[^>]+>\s*'
    return re.sub(pattern, '', sentence)

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device {device}')    

    if torch.cuda.is_available():
        logger.info(f'GPU machine name: {torch.cuda.get_device_name(0)}')
        logger.info(f'SSH machine name: {socket.gethostname()}')
    
    seed_value = 53
    setup_seed(seed_value)

    langs = ['en', 'hi', 'mag', 'ml', 'mr', 'or', 'pa', 'te', 'ur']

    logger.info("Loading data...")
    train_df, dev_df, test_df_dict = load_data(args.data_path, langs)

    if args.task == 'pos_to_neg':
        src = 'POSITIVE'
        trg = 'NEGATIVE'
        test_src = 'POSITIVE'
        test_print_src = 'POSITIVE'
        test_trg = 'NEGATIVE'
    else:
        src = 'NEGATIVE'
        trg = 'POSITIVE'
        test_src = 'NEGATIVE'
        test_print_src = 'NEGATIVE'
        test_trg = 'POSITIVE'

    shutil.rmtree('facebook', ignore_errors=True)

    logger.info("Initializing tokenizer...")
    tokenizer = MBart50TokenizerFast.from_pretrained(args.model_name)

    special_tokens_dict = {'additional_special_tokens': ['<en>', '<hi>', '<mag>', '<ml>', '<mr>', '<or>', '<pa>', '<te>', '<ur>']}
    tokenizer.add_special_tokens(special_tokens_dict)

    train_src_encodings = tokenizer(train_df[src].values.tolist(), truncation=True, padding=True, max_length=128)
    train_trg_encodings = tokenizer(train_df[trg].values.tolist(), truncation=True, padding=True, max_length=128)

    dev_src_encodings = tokenizer(dev_df[src].values.tolist(), truncation=True, padding=True, max_length=128)
    dev_trg_encodings = tokenizer(dev_df[trg].values.tolist(), truncation=True, padding=True, max_length=128)

    train_dataset = CreateDataset(train_src_encodings, train_trg_encodings)
    dev_dataset = CreateDataset(dev_src_encodings, dev_trg_encodings)

    logger.info("Initializing model...")
    
    model = MBartForConditionalGeneration.from_pretrained(args.model_name)
    
    model.resize_token_embeddings(len(tokenizer))
    
    model.to(device)

    args_train = Seq2SeqTrainingArguments(
        args.model_name,
        evaluation_strategy="epoch",
        learning_rate=1e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=0.01,
        save_total_limit=1,
        save_strategy='epoch',
        load_best_model_at_end=True,
        num_train_epochs=args.epochs,
        predict_with_generate=True,
        fp16=True
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model,
        args_train,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    logger.info("Training the model...")
    trainer.train()

    logger.info("Evaluating the model...")
    trainer.evaluate()

    for lang in langs:
        test_df = test_df_dict[lang]
        pred = []
        for idx in range(len(test_df[test_src].values.tolist())):
            src_sentence = test_df[test_src].values.tolist()[idx]
            pred.append(gen(src_sentence, tokenizer, model))
        
        # output = {
        #     'src': test_df[test_print_src].values.tolist(),
        #     'trg': test_df[test_trg].values.tolist(),
        #     'pred': pred
        # }
        
        clean_src_sentences = [remove_tags(sentence) for sentence in test_df[test_print_src].values.tolist()]
        clean_trg_sentences = [remove_tags(sentence) for sentence in test_df[test_trg].values.tolist()]
        
        output = {
            'src': clean_src_sentences,
            'trg': clean_trg_sentences,
            'pred': pred
        }
        
        output_df = pd.DataFrame(output)
        output_file = args.output_dir + args.task + '-' + lang + '-' + args.methodology + '.csv'
        output_df.to_csv(output_file, index=False)

        tst_evaluator = TSTEvaluator(
            task=args.task,
            lang=lang,
            output_file=output_file,
            methodology=args.methodology
        )
    
        logger.info(f'Starting evaluation process for lang: {lang}')
        tst_evaluator.set_seed(53)
        accuracy, similarity_score, bleu, fluency_scores_mgpt, fluency_scores_xlmr = tst_evaluator.evaluate()
        
        accuracy_text = f", Sentiment Accuracy: {accuracy:.2f}" if accuracy is not None else ", Sentiment Accuracy: None"
        similarity_score_text = f", Similarity: {similarity_score:.2f}" if similarity_score is not None else ", Similarity: None"
        bleu_text = f", Bleu Score: {bleu:.2f}" if bleu is not None else ", Bleu Score: None"
    
        fluency_mgpt_text = ", ".join([f"MGPT {key}: {value:.2f}" if value is not None else f"MGPT {key}: None" for key, value in fluency_scores_mgpt.items()])
        fluency_xlmr_text = ", ".join([f"XLM-R {key}: {value:.2f}" if value is not None else f"XLM-R {key}: None" for key, value in fluency_scores_xlmr.items()])
        
        logger.info(f"Results Summary for Language: {lang}, Methodology: {args.methodology}, task: {args.task}, Batch Size: {args.batch_size} - {accuracy_text}{similarity_score_text}{bleu_text},{fluency_mgpt_text}, {fluency_xlmr_text}")
    
        logger.info('=' * 50 + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Joint model, all the languages data trained together')
    parser.add_argument('--data_path', type=str, default='../data/', help='path to data directory')
    parser.add_argument('--task', type=str, choices=['pos_to_neg', 'neg_to_pos'], required=True, help='task type: pos_to_neg or neg_to_pos')
    parser.add_argument('--model_name', type=str, default='facebook/mbart-large-50', help='pretrained model name')
    parser.add_argument('--output_dir', type=str, default='../output/', help='output directory for saving results')
    parser.add_argument('--methodology', type=str, default='joint', help='methodology used for testing')
    parser.add_argument('--batch_size', type=int, default=3, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=30, help='number of training epochs')
    args = parser.parse_args()

    logger.info("Command-line arguments:")
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")

    start_time = time.time()
    
    main(args)

    end_time = time.time()
    time_taken_seconds = end_time - start_time
    time_taken_formatted = time.strftime('%H:%M:%S', time.gmtime(time_taken_seconds))

    logger.info(f"Time taken for methodology: {args.methodology} - {time_taken_formatted}")