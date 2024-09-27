import shutil
import random
import torch
import socket
import os
import argparse
import time
# import coloredlogs
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from my_logger import logger
from automatic_eval import TSTEvaluator
from transformers import TrainerCallback
from torch.utils.data import DataLoader, TensorDataset
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, MBartConfig

class CustomLoggingCallback(TrainerCallback):
    def __init__(self):
        self.best_eval_loss = float('inf')
        self.best_epoch = -1

    def on_evaluate(self, args, state, control, **kwargs):
        train_loss = None
        eval_loss = None
        
        for item in reversed(state.log_history):
            if "loss" in item:
                train_loss = item["loss"]
                break
        
        for item in reversed(state.log_history):
            if "eval_loss" in item:
                eval_loss = item["eval_loss"]
                break
        
        if train_loss is not None and eval_loss is not None:
            logger.info(f"Epoch {state.epoch}: Train Loss - {train_loss}, Eval Loss - {eval_loss}")
            if eval_loss < self.best_eval_loss:
                self.best_eval_loss = eval_loss
                self.best_epoch = state.epoch
        elif train_loss is not None:
            logger.info(f"Epoch {state.epoch}: Train Loss - {train_loss}, No evaluation loss available")
        else:
            logger.info(f"Epoch {state.epoch}: No training or evaluation loss available")


class TextStyleTransfer:
    def __init__(self, model_name, batch_size, epochs, task, lang, src_lang_code, trg_lang_code, methodology, src, trg, test_src, test_trg, test_print_src, output_file):
        self.model_name = model_name
        self.batch_size = batch_size
        self.epochs = epochs
        self.task = task
        self.lang = lang
        self.src_lang_code = src_lang_code
        self.trg_lang_code = trg_lang_code
        self.methodology = methodology
        self.src = src 
        self.trg = trg 
        self.test_src = test_src
        self.test_trg = test_trg
        self.test_print_src = test_print_src
        self.output_file = output_file
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def load_data(self, lang):
        train_df = pd.read_csv(f'../data/{lang}_train_025.csv')
        dev_df = pd.read_csv(f'../data/{lang}_dev_025.csv')
        test_df = pd.read_csv(f'../data/{lang}_test_025.csv')
        return train_df, dev_df, test_df

    def create_dataset(self, src_encodings, trg_encodings):
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

        return CreateDataset(src_encodings, trg_encodings)
        

    def save_output(self, df):
        output_csv = f'{self.output_file}'
        df.to_csv(output_csv, index=False)

    def main(self):
        self.set_seed(53)

        train_df, dev_df, test_df = self.load_data(self.lang)
        tokenizer = MBart50TokenizerFast.from_pretrained(self.model_name, src_lang=self.src_lang_code, tgt_lang=self.trg_lang_code)

        train_src_encodings = tokenizer(train_df[self.src].values.tolist(), truncation=True, padding=True, max_length=128)
        train_trg_encodings = tokenizer(train_df[self.trg].values.tolist(), truncation=True, padding=True, max_length=128)

        dev_src_encodings = tokenizer(dev_df[self.src].values.tolist(), truncation=True, padding=True, max_length=128)
        dev_trg_encodings = tokenizer(dev_df[self.trg].values.tolist(), truncation=True, padding=True, max_length=128)

        train_dataset = self.create_dataset(train_src_encodings, train_trg_encodings)
        dev_dataset = self.create_dataset(dev_src_encodings, dev_trg_encodings)
        
        model = MBartForConditionalGeneration.from_pretrained(self.model_name)
        model.to(self.device)
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

        args = Seq2SeqTrainingArguments(
            self.model_name,
            evaluation_strategy="epoch",
            logging_strategy="epoch",
            logging_steps=len(train_df),
            learning_rate=1e-5,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            weight_decay=0.01,
            save_total_limit=1,
            save_strategy='epoch',
            load_best_model_at_end=True,
            num_train_epochs=self.epochs,
            predict_with_generate=True,
            fp16=True
        )

        loggingCallback = CustomLoggingCallback()
        
        trainer = Seq2SeqTrainer(
            model,
            args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            callbacks=[loggingCallback]
        )
        
        trainer.train()
        trainer.evaluate()

        logger.info(f"Best evaluation loss found at epoch {loggingCallback.best_epoch} for Language: {self.lang}, Methodology: {self.methodology}, task: {self.task}, Batch Size: {self.batch_size}")

        pred = []
        for ip in test_df[self.test_src].values.tolist():
            src_tknz = tokenizer(ip, truncation=True, padding=True, max_length=128, return_tensors='pt')
            src_tknz = {k: v.to(self.device) for k, v in src_tknz.items()}
            generated_ids = model.generate(src_tknz["input_ids"], max_length=128)
            pred.append(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])
        
        output = {
            'src': test_df[self.test_print_src].values.tolist(),
            'trg': test_df[self.test_trg].values.tolist(),
            'pred': pred
        }

        logger.info(self.task)
        logger.info('-' * 10)
        logger.info('\n')
    
        for idx in range(5):
            logger.info('src: %s', test_df[self.test_print_src].values.tolist()[idx])
            logger.info('trg: %s', test_df[self.test_trg].values.tolist()[idx])
            logger.info('pred: %s', pred[idx])
            logger.info('\n')
        
        output_df = pd.DataFrame(output)
        
        self.save_output(output_df)
        logger.info('Text Style Transfer process completed.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multilingual Text Style Transfer')
    parser.add_argument('--model_name', type=str, default='facebook/mbart-large-50', help='Pretrained model name')
    parser.add_argument('--batch_size', type=int, default=3, help='Batch size')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--task', type=str, required=True, help='Task (pos_to_neg or neg_to_pos')
    parser.add_argument('--lang', type=str, required=True, help='Language (bn or en)')
    parser.add_argument('--src_lang_code', type=str, required=True, help='')
    parser.add_argument('--trg_lang_code', type=str, required=True, help='')
    parser.add_argument('--methodology', type=str, required=True, help='Methodology (parallel, ae, bt, ae_mask, or bt_mask)')
    parser.add_argument('--src', type=str, required=True, help='')
    parser.add_argument('--trg', type=str, required=True, help='')
    parser.add_argument('--test_src', type=str, required=True, help='')
    parser.add_argument('--test_trg', type=str, required=True, help='')
    parser.add_argument('--test_print_src', type=str, required=True, help='')
    parser.add_argument('--output_file', type=str, required=True, help='Output file')
    args = parser.parse_args()

    if args.src_lang_code == 'None':
        args.src_lang_code = None

    if args.trg_lang_code == 'None':
        args.trg_lang_code = None

    logger.info("Command-line arguments:")
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device {device}')    

    if torch.cuda.is_available():
        logger.info(f'GPU machine name: {torch.cuda.get_device_name(0)}')
        logger.info(f'SSH machine name: {socket.gethostname()}')

    logger.info(f'Starting TST process.')
    shutil.rmtree('facebook', ignore_errors=True)

    start_time = time.time()
    
    text_style_transfer = TextStyleTransfer(args.model_name, args.batch_size, args.epochs, args.task, args.lang, args.src_lang_code, args.trg_lang_code, args.methodology, args.src, args.trg, args.test_src, args.test_trg, args.test_print_src, args.output_file)
    
    text_style_transfer.main()
    
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

    end_time = time.time()
    time_taken_seconds = end_time - start_time
    time_taken_formatted = time.strftime('%H:%M:%S', time.gmtime(time_taken_seconds))

    logger.info(f"Time taken for language: {args.lang}, methodology: {args.methodology}, task: {args.task} - {time_taken_formatted}")

    logger.info('=' * 50 + '\n')