import argparse
import logging
import pandas as pd
import numpy as np
import random
import torch
import socket
# from transformers import BertTokenizer
# from transformers import BertForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers_interpret import SequenceClassificationExplainer
from sklearn.metrics import f1_score

class MaskStyle:
    def __init__(self, model_name, data_path, threshold, lang):
        self.model_name = model_name
        self.data_path = data_path
        self.threshold = threshold
        self.lang = lang
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def set_seed(self, seed_value):
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)

    # def merge_contractions(self, attributions):
    #     merged_attributions = []
    #     i = 0
    #     while i < len(attributions):
    #         word, score = attributions[i]
    #         if i + 2 < len(attributions) and attributions[i + 1][0] == "'" and attributions[i + 2][0] in ["t", "re", "ve", "s", "m", "ll", "d"]:
    #             merged_word = word + attributions[i + 1][0] + attributions[i + 2][0]
    #             merged_score = score + attributions[i + 1][1] + attributions[i + 2][1]
    #             merged_attributions.append((merged_word, merged_score))
    #             i += 3
    #         else:
    #             merged_attributions.append((word, score))
    #             i += 1
    #     return merged_attributions

    def calculate_attributions(self, cls_explainer, sentences):
        masked_sentences = []
        for sentence in sentences:
            
            if isinstance(sentence, bytes):
                sentence = sentence.decode('utf-8')
            # logging.info(f'sentence: {sentence}')
            
            # attributions = cls_explainer(sentence)
            # word_attributions = []
            # word = ""
            # word_score = 0.0
            # for token, score in attributions:
            #     if token.startswith("_"):
            #         word += token[2:]
            #         word_score += score
            #     else:
            #         if word:
            #             word_attributions.append((word, word_score))
            #         word = token
            #         word_score = score
            # if word:
            #     word_attributions.append((word, word_score))
            # if self.lang == 'en':
            #     merged_attributions = self.merge_contractions(word_attributions)
            # else:
            #     merged_attributions = word_attributions

            word_attributions = cls_explainer(sentence)
            processed_word_attributions = []
            current_word = ''
            current_score = 0.0
            for token, score in word_attributions:
                if token in ['<s>', '</s>']:
                    continue
                # If the token starts with '▁', it's the start of a new word
                if token.startswith('▁'):
                    # If there's a previous word, add it to the processed word attributions
                    if current_word:
                        processed_word_attributions.append((current_word, current_score))
                    # Reset the current word and score
                    current_word = token[1:]
                    current_score = score
                else:
                    # If the token doesn't start with '▁', it's part of the current word
                    current_word += token
                    # Accumulate score for tokens part of the same word
                    current_score += score
            # Add the last word if any
            if current_word:
                processed_word_attributions.append((current_word, current_score))
                
            masked_sentence = []
            for word, score in processed_word_attributions:
                if score >= self.threshold:
                    masked_sentence.append("<mask>")
                elif word not in ['<s>', '</s>']:
                    masked_sentence.append(word)
            masked_sentences.append(" ".join(masked_sentence))
        return masked_sentences

    def mask_sentences(self, df, column_name):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name,
                                                      num_labels=2,
                                                      output_attentions=False,
                                                      output_hidden_states=False)
        model.to(self.device)

        # Fix for xlm-roberta like models: https://github.com/cdpierse/transformers-interpret/issues/123#issuecomment-1668096140
        model.load_state_dict(torch.load('../classifier/models_xlmr/' + self.lang + '_best.model'))
        
        cls_explainer = SequenceClassificationExplainer(model, tokenizer)
        
        # Fix for predicted probs computed not correct: https://github.com/cdpierse/transformers-interpret/issues/65#issuecomment-1165662214
        cls_explainer.accepts_position_ids = False
        sentences = df[column_name].tolist()
        masked_sentences = self.calculate_attributions(cls_explainer, sentences)
        new_column_name = f"{column_name}_MASK"
        df[new_column_name] = masked_sentences
        return df

    def process_dataframes(self):
        for df_name, df in dataframes.items():
            for column_name in ['POSITIVE', 'NEGATIVE', 'POSITIVE_TR', 'NEGATIVE_TR']:
                df = self.mask_sentences(df, column_name)
            df.to_csv(f'../data/{df_name}_{str(self.threshold).replace(".", "")}.csv', index = False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mask Style")
    parser.add_argument("--model_name", type=str, default="xlm-roberta-base", help="Model name")
    parser.add_argument("--data_path", type=str, default="../data/", help="Data path")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold")
    parser.add_argument("--lang", type=str, required=True, help="Language")
    args = parser.parse_args()

    logging.basicConfig(filename='style_masking.log', level=logging.INFO)

    # Log the command-line arguments
    logging.info("Command-line arguments:")
    for arg, value in vars(args).items():
        logging.info(f"{arg}: {value}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Device {device}')    

    if torch.cuda.is_available():
        logging.info(f'GPU machine name: {torch.cuda.get_device_name(0)}')
        logging.info(f'SSH machine name: {socket.gethostname()}')
    
    mask_style = MaskStyle(args.model_name, args.data_path, args.threshold, args.lang)
    mask_style.set_seed(53)
    dataframes = {
        f'{args.lang}_train': pd.read_csv(f'{args.data_path}{args.lang}_train.csv'),
        f'{args.lang}_dev': pd.read_csv(f'{args.data_path}{args.lang}_dev.csv'),
        f'{args.lang}_test': pd.read_csv(f'{args.data_path}{args.lang}_test.csv')
    }

    # # Fill NaN values in all dataframes with 'not a valid string'
    # fill_value = 'not a valid string'
    
    # for df_name, df in dataframes.items():
    #     dataframes[df_name] = df.fillna(fill_value)
    
    mask_style.process_dataframes()
