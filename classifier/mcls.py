import argparse
import pandas as pd
import numpy as np
import random
import socket
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import BertTokenizer, BertForSequenceClassification 
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, XLMRobertaConfig
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import time
from tqdm import tqdm
import logging

class MultilingualSentimentClassifier:
    def __init__(self, device, data_path, model_path, language, model_name, batch_size, epochs, early_stop, early_stop_limit, train_together, langs):
        self.device = device
        self.data_path = data_path
        self.model_path = model_path
        self.language = language
        self.model_name = model_name
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stop = early_stop
        self.early_stop_limit = early_stop_limit
        self.train_together = train_together
        self.langs = langs

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def load_data(self, data_path, language):
        train_df = pd.read_csv(data_path + language + '_train.csv')
        dev_df = pd.read_csv(data_path + language + '_dev.csv')
        test_df = pd.read_csv(data_path + language + '_test.csv')
        return train_df, dev_df, test_df

    def load_data_all(self, data_path, langs):
        train_df_dict = {}
        dev_df_dict = {}
        test_df_dict = {}
    
        for lang in langs:
            train_df = pd.read_csv(f'{data_path}{lang}_train.csv')
            dev_df = pd.read_csv(f'{data_path}{lang}_dev.csv')
            test_df = pd.read_csv(f'{data_path}{lang}_test.csv')
    
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
        test_df = pd.concat(test_df_dict.values(), ignore_index=True)

        return train_df, dev_df, test_df, train_df_dict, dev_df_dict, test_df_dict

    def preprocess_data(self, df):
        pos = df['POSITIVE'].to_list()
        neg = df['NEGATIVE'].to_list()
        
        pos_label = [1] * len(df)
        neg_label = [0] * len(df)
        
        df_cls = pd.DataFrame(list(zip(pos + neg, pos_label + neg_label)), columns=['Text', 'Label'])
        
        return df_cls.sample(frac=1)

    def encode_data(self, tokenizer, df_cls):
        encoded_data = tokenizer.batch_encode_plus(
            df_cls['Text'].tolist(),
            add_special_tokens=True,
            return_attention_mask=True,
            max_length=128,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
    
        input_ids = encoded_data['input_ids']
        attention_masks = encoded_data['attention_mask']
        labels = torch.tensor(df_cls.Label.values)
    
        return input_ids, attention_masks, labels

    def create_datasets(self, input_ids_train, attention_masks_train, labels_train, input_ids_val, attention_masks_val, labels_val, input_ids_test, attention_masks_test, labels_test):
        dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
        dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)
        dataset_test = TensorDataset(input_ids_test, attention_masks_test, labels_test)

        return dataset_train, dataset_val, dataset_test

    def initialize_model(self, model_name):
        # model = BertForSequenceClassification.from_pretrained(model_name,
        #                                                     num_labels=2,
        #                                                     output_attentions=False,
        #                                                     output_hidden_states=False)
        model = AutoModelForSequenceClassification.from_pretrained(model_name,
                                                            num_labels=2,
                                                            output_attentions=False,
                                                            output_hidden_states=False)
        return model

    def evaluate(self, dataloader_val, model):
        model.eval()

        loss_val_total = 0
        predictions, true_vals = [], []

        for batch in dataloader_val:
            batch = tuple(b.to(self.device) for b in batch)

            inputs = {
                'input_ids': batch[0],
                'attention_mask': batch[1],
                'labels': batch[2],
            }

            with torch.no_grad():
                outputs = model(**inputs)

            loss = outputs[0]
            logits = outputs[1]
            loss_val_total += loss.item()

            logits = logits.detach().cpu().numpy()
            label_ids = inputs['labels'].cpu().numpy()
            predictions.append(logits)
            true_vals.append(label_ids)

        loss_val_avg = loss_val_total / len(dataloader_val)

        predictions = np.concatenate(predictions, axis=0)
        true_vals = np.concatenate(true_vals, axis=0)

        return loss_val_avg, predictions, true_vals

    def train_and_evaluate(self, train_dataloader, val_dataloader, model, optimizer, scheduler, device):
        best_val_loss = float('inf')
        early_stop_cnt = 0
        best_epoch = 0

        for epoch in range(1, self.epochs + 1):
            model.train()

            start_time = time.time()
            
            loss_train_total = 0
    
            progress_bar = tqdm(train_dataloader, desc='Training', leave=False, disable=False)
            for batch in progress_bar:
                model.zero_grad()
    
                batch = tuple(b.to(device) for b in batch)
    
                inputs = {
                    'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'labels': batch[2],
                }
    
                outputs = model(**inputs)
    
                loss = outputs[0]
                loss_train_total += loss.item()
                loss.backward()
    
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
                optimizer.step()
                scheduler.step()
    
            loss_train_avg = loss_train_total / len(train_dataloader)
            
            val_loss, predictions, true_vals = self.evaluate(val_dataloader, model)

            if val_loss < best_val_loss:
                early_stop_cnt = 0
            elif val_loss >= best_val_loss:
                early_stop_cnt += 1

            if val_loss < best_val_loss:
                early_stop_cnt = 0
                best_val_loss = val_loss
                best_epoch = epoch
                
                torch.save(model.state_dict(), self.model_path+self.language + '_best.model')
            
            elif val_loss >= best_val_loss:
                early_stop_cnt += 1

            else:
                early_stop_cnt += 1

            val_f1 = self.f1_score_func(predictions, true_vals)

            end_time = time.time()
            epoch_mins, epoch_secs = self.epoch_time(start_time, end_time)

            logging.info(f'Epoch: {epoch:02} | Time: {epoch_mins}m {epoch_secs}s')
            logging.info(f'\tTrain Loss: {loss_train_avg:.3f}')
            logging.info(f'\t Val. Loss: {val_loss:.3f}')
            logging.info(f'\t Val. f1 score: {val_f1:.3f}')

            if self.early_stop:
                if early_stop_cnt == self.early_stop_limit:
                    logging.info('Early Stopping...')
                    break
        
        logging.info(f'Best epoch: {best_epoch} | Best validation loss: {best_val_loss:.3f}')

    def f1_score_func(self, preds, labels):
        preds_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return f1_score(labels_flat, preds_flat, average='weighted')

    def accuracy_per_class(self, preds, labels):
        preds_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        
        class_accuracies = []
    
        for label in np.unique(labels_flat):
            y_preds = preds_flat[labels_flat == label]
            y_true = labels_flat[labels_flat == label]
            
            accuracy = len(y_preds[y_preds == label]) / len(y_true)
            class_accuracies.append(accuracy)
        
            logging.info(f'Class: {label}')
            logging.info(f'Accuracy: {len(y_preds[y_preds == label])}/{len(y_true)}, {accuracy * 100:.2f}% \n')
        
        average_accuracy = np.mean(class_accuracies)
        logging.info(f'Average Accuracy: {average_accuracy * 100:.2f}%')

    def epoch_time(self, start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

def main(args):
    logging.basicConfig(filename='mcls.log', level=logging.INFO)
    logging.info(f'Starting Sentiment Classifier process for {args.language}')

    # Log the command-line arguments
    logging.info("Command-line arguments:")
    for arg, value in vars(args).items():
        logging.info(f"{arg}: {value}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Device {device}')    

    if torch.cuda.is_available():
        logging.info(f'GPU machine name: {torch.cuda.get_device_name(0)}')
        logging.info(f'SSH machine name: {socket.gethostname()}')

    
    multilingual_sentiment_classifier = MultilingualSentimentClassifier(device, args.data_path, args.model_path, args.language, args.model_name, args.batch_size, args.epochs, args.early_stop, args.early_stop_limit, args.train_together, args.langs)

    multilingual_sentiment_classifier.set_seed(args.seed)

    if(args.train_together):
        train_df, dev_df, test_df, train_df_dict, dev_df_dict, test_df_dict = multilingual_sentiment_classifier.load_data_all(args.data_path, args.langs)
    else:
        train_df, dev_df, test_df = multilingual_sentiment_classifier.load_data(args.data_path, args.language)
    
    # train_df = train_df.fillna('not a valid string')
    # dev_df = dev_df.fillna('not a valid string')
    # test_df = test_df.fillna('not a valid string')
    
    # train_df_cls, dev_df_cls, test_df_cls = multilingual_sentiment_classifier.preprocess_data(train_df, dev_df, test_df)
    train_df_cls = multilingual_sentiment_classifier.preprocess_data(train_df)
    dev_df_cls = multilingual_sentiment_classifier.preprocess_data(dev_df)
    test_df_cls = multilingual_sentiment_classifier.preprocess_data(test_df)

    # tokenizer = BertTokenizer.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    (input_ids_train, attention_masks_train, labels_train) = multilingual_sentiment_classifier.encode_data(tokenizer, train_df_cls)
    (input_ids_val, attention_masks_val, labels_val) = multilingual_sentiment_classifier.encode_data(tokenizer, dev_df_cls)
    (input_ids_test, attention_masks_test, labels_test) = multilingual_sentiment_classifier.encode_data(tokenizer, test_df_cls)

    dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
    dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)
    dataset_test = TensorDataset(input_ids_test, attention_masks_test, labels_test)

    train_dataloader = DataLoader(dataset_train, sampler=RandomSampler(dataset_train), batch_size=args.batch_size)
    val_dataloader = DataLoader(dataset_val, sampler=SequentialSampler(dataset_val), batch_size=args.batch_size)
    test_dataloader = DataLoader(dataset_test, sampler=SequentialSampler(dataset_test), batch_size=args.batch_size)

    # Define dataloaders for validation and test sets for each language
    dataloaders_validation = {}
    dataloaders_test = {}
    
    if(args.train_together):
        for lang in args.langs:
    
            dev_df_lang_cls = multilingual_sentiment_classifier.preprocess_data(dev_df_dict[lang])
            test_df_lang_cls = multilingual_sentiment_classifier.preprocess_data(test_df_dict[lang])
    
            (input_ids_val_lang, attention_masks_val_lang, labels_val_lang) = multilingual_sentiment_classifier.encode_data(tokenizer, dev_df_lang_cls)
            (input_ids_test_lang, attention_masks_test_lang, labels_test_lang) = multilingual_sentiment_classifier.encode_data(tokenizer, test_df_lang_cls)
    
            dataset_val_lang = TensorDataset(input_ids_val_lang, attention_masks_val_lang, labels_val_lang)
            dataset_test_lang = TensorDataset(input_ids_test_lang, attention_masks_test_lang, labels_test_lang)
            
            dataloader_validation_lang = DataLoader(dataset_val_lang,
                                                    sampler=SequentialSampler(dataset_val_lang),
                                                    batch_size=args.batch_size)
            dataloader_test_lang = DataLoader(dataset_test_lang,
                                              sampler=SequentialSampler(dataset_test_lang),
                                              batch_size=args.batch_size)
            
            dataloaders_validation[lang] = dataloader_validation_lang
            dataloaders_test[lang] = dataloader_test_lang
    
    
    model = multilingual_sentiment_classifier.initialize_model(args.model_name)

    model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, eps=1e-8)

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * args.epochs)

    multilingual_sentiment_classifier.train_and_evaluate(train_dataloader, val_dataloader, model, optimizer, scheduler, device)

    model.load_state_dict(torch.load(args.model_path+args.language+'_best.model'))

    
    logging.info("Accuracy on Dev Data:")

    _, predictions, true_vals = multilingual_sentiment_classifier.evaluate(val_dataloader, model)
    
    multilingual_sentiment_classifier.accuracy_per_class(predictions, true_vals)

   
    logging.info("Accuracy on Test Data:")
    _, predictions, true_vals = multilingual_sentiment_classifier.evaluate(test_dataloader, model)
    
    multilingual_sentiment_classifier.accuracy_per_class(predictions, true_vals)

    if(args.train_together):
        logging.info("Evaluate on validation set for each language and print accuracy per class:")
        for lang in args.langs:
            logging.info(f'{lang}:')
            _, predictions, true_vals = multilingual_sentiment_classifier.evaluate(dataloaders_validation[lang], model)
            multilingual_sentiment_classifier.accuracy_per_class(predictions, true_vals)
        
        logging.info("Evaluate on test set for each language and print accuracy per class:")
        for lang in args.langs:
            logging.info(f'{lang}:')
            _, predictions, true_vals = multilingual_sentiment_classifier.evaluate(dataloaders_test[lang], model)
            multilingual_sentiment_classifier.accuracy_per_class(predictions, true_vals)

    logging.info(f'Sentiment classifier process completed for {args.language}')

    logging.info('=' * 50 + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multilingual Sentiment Classifier')
    parser.add_argument('--seed', type=int, default=53, help='Random seed')
    parser.add_argument('--data_path', type=str, default='../data/', help='Data path for data files')
    parser.add_argument('--model_path', type=str, default='models_xlmr/', help='Data path for data files')
    parser.add_argument('--language', type=str, default='all', help='Language for data')
    parser.add_argument('--model_name', type=str, default='xlm-roberta-base', help='Pretrained model name')
    parser.add_argument('--batch_size', type=int, default=3, help='Batch size')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument("--early_stop", action="store_true", help="Set this flag to enable early_stop")
    parser.add_argument('--early_stop_limit', type=int, default=5, help='Early stopping limit')
    parser.add_argument("--train_together", action="store_true", help="Set this flag to train using the languages together")
    parser.add_argument('--langs', nargs='+', default=['en', 'hi', 'mag', 'ml', 'mr', 'or', 'pa', 'te', 'ur'], help='List of languages')

    args = parser.parse_args()
    
    main(args)