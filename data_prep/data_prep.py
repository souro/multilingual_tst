import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import random
import numpy as np
import time
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import logging
import socket

class DataPreparation:
    def __init__(self, seed, data_path, lang, trns_intrmdt_lang, bck_trns_src_lang, save_path, bt, en_to_others_trns, en_to_others_trns_langs):
        self.seed = seed
        self.data_path = data_path
        self.lang = lang
        self.trns_intrmdt_lang = trns_intrmdt_lang
        self.bck_trns_src_lang = bck_trns_src_lang
        self.save_path = save_path
        self.bt = bt
        self.en_to_others_trns = en_to_others_trns
        self.en_to_others_trns_langs = en_to_others_trns_langs

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def load_data(self):
        df_neg_to_pos = pd.read_csv(self.data_path + self.lang + "_yelp_reference-0.csv")
        df_pos_to_neg = pd.read_csv(self.data_path + self.lang + "_yelp_reference-1.csv")
        df = pd.concat([df_neg_to_pos, df_pos_to_neg], ignore_index=True)
        return df

    def split_data(self, df):
        train, temp = train_test_split(df, test_size=600, random_state=self.seed)
        dev, test = train_test_split(temp, test_size=500, random_state=self.seed)
        return train, dev, test

    def translate(self, df, column, lan_code, tokenizer, model, new_column_name):
        start_time = time.time()

        inputs = tokenizer(df[column].to_list(), return_tensors="pt", padding=True, truncation=True, max_length=30).to(model.device)
        translated_tokens = model.generate(
            **inputs, forced_bos_token_id=tokenizer.lang_code_to_id[lan_code], max_length=30
        )
        translated_texts = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)

        df[new_column_name] = translated_texts

        end_time = time.time()
        time_taken_seconds = end_time - start_time
        time_taken_formatted = time.strftime('%H:%M:%S', time.gmtime(time_taken_seconds))
        logging.info(f"Translation processed {new_column_name} for {lan_code} of {len(df)} rows in {time_taken_formatted}")

        return df
    
    def save_data(self, name, df):
        df.to_csv(self.save_path + self.lang + "_"+name+".csv", index=False)

    def process_data(self):
        self.set_seed(self.seed)
        df = self.load_data()
        train, dev, test = self.split_data(df)

        df_names = {'train': train, 'dev': dev, 'test': test}
        for key, val in df_names.items():
            self.save_data(key, val)

        if self.bt:
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-3.3B")
            model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-3.3B").to("cuda" if torch.cuda.is_available() else "cpu")
    
            for key, val in df_names.items():
                for column in ['POSITIVE', 'POSITIVE_TR', 'NEGATIVE', 'NEGATIVE_TR']:
                    if('_TR' not in column):
                        val = self.translate(val, column, self.trns_intrmdt_lang, tokenizer, model, f'{column}_TR')

                        if self.lang == 'en' and self.en_to_others_trns:
                            for other_lang in self.en_to_others_trns_langs:
                                val = self.translate(val, column, other_lang, tokenizer, model, f'{column}_{other_lang}')
                    else:
                        val = self.translate(val, column, self.bck_trns_src_lang, tokenizer, model, f'{column}_TR')
                
                self.save_data(key, val)

def main():
    parser = argparse.ArgumentParser(description='Data Preparation for Multilingual TST')
    parser.add_argument('--seed', type=int, default=53, help='Random seed')
    parser.add_argument('--data_path', type=str, required=True, help='Path to data')
    parser.add_argument('--lang', type=str, default='en', help='Language code')
    parser.add_argument('--save_path', type=str, default='../data/', help='Path to save processed data')
    
    parser.add_argument("--bt", action="store_true", help="Set this flag to prepare translation and back-translation of the data")
    parser.add_argument('--trns_intrmdt_lang', type=str, default=None, help='Language code')
    parser.add_argument('--bck_trns_src_lang', type=str, default=None, help='Language code')

    parser.add_argument('--en_to_others_trns', action="store_true", help="Set this flag to enable translation from English to other languages")
    parser.add_argument('--en_to_others_trns_langs', type=str, default="mag_Deva mar_Deva pan_Guru mal_Mlym ory_Orya hin_Deva urd_Arab ben_Beng tel_Telu", help="Provide list of language codes separated by spaces to translate from English")

    args = parser.parse_args()

    if args.bt:
        if args.trns_intrmdt_lang is None or args.bck_trns_src_lang is None:
            parser.error("--trns_intrmdt_lang and --bck_trns_src_lang are required when --bt is set.")

    logging.basicConfig(filename='data_prep.log', level=logging.INFO)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Device {device}')    

    if torch.cuda.is_available():
        logging.info(f'GPU machine name: {torch.cuda.get_device_name(0)}')
        logging.info(f'SSH machine name: {socket.gethostname()}')
    
    logging.info("Command-line arguments:")
    for arg, value in vars(args).items():
        logging.info(f"{arg}: {value}")

    logging.info('Starting data preparation process.')

    en_to_others_trns_langs = None
    if args.lang == 'en' and args.en_to_others_trns:
        if not args.en_to_others_trns_langs:
            parser.error("--en_to_others requires specifying --languages.")
        else:
            en_to_others_trns_langs = args.en_to_others_trns_langs.split()
    
    data_preparation = DataPreparation(args.seed, args.data_path, args.lang, args.trns_intrmdt_lang, args.bck_trns_src_lang, args.save_path, args.bt, args.en_to_others_trns, en_to_others_trns_langs)
    data_preparation.process_data()
    
    logging.info('Data preparation process completed.')

if __name__ == "__main__":
    main()
