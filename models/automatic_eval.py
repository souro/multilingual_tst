import argparse
import torch
import random
import pandas as pd
import numpy as np
import os
import traceback
import socket
import evaluate
from my_logger import logger
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForMaskedLM
from transformers import GPT2Tokenizer, GPT2LMHeadModel

class TSTEvaluator:
    def __init__(self, task, lang, output_file, methodology):
        self.task = task
        self.lang = lang
        self.output_file = output_file
        self.methodology = methodology
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def set_seed(self, seed_value):
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)

    def sentiment_accuracy(self, text, trg_label, lang):
        try:
            cls_model_name = "xlm-roberta-base"
            cls_tokenizer = AutoTokenizer.from_pretrained(cls_model_name)
            cls_model = AutoModelForSequenceClassification.from_pretrained(cls_model_name,
                                                                           num_labels=2,
                                                                           output_attentions=False,
                                                                           output_hidden_states=False)
            cls_model.to(self.device)
            cls_model.load_state_dict(torch.load(f'../classifier/models_xlmr/{lang}_best.model'))

            inputs = cls_tokenizer(text, padding=True, truncation=True, return_tensors="pt")
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)

            with torch.no_grad():
                outputs = cls_model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits

            predicted_labels = logits.argmax(dim=1).cpu().numpy()

            accuracy = accuracy_score(trg_label, predicted_labels)

            return accuracy
        except Exception as e:
            logger.error(f"Error in sentiment_accuracy: {e}")
            logger.error(traceback.format_exc())
            return None

    def similarity(self, text1, text2, lang):
        try:
            sim_model = SentenceTransformer('sentence-transformers/LaBSE')
            sim_model.to(self.device)

            sim_scores = list()
            for idx, _ in enumerate(text1):
                sentence_embedding1 = sim_model.encode(text1[idx])
                sentence_embedding2 = sim_model.encode(text2[idx])
                sim_score = cosine_similarity([sentence_embedding1], [sentence_embedding2])
                sim_scores.append(sim_score[0][0])

            return sum(sim_scores) / len(sim_scores)
        except Exception as e:
            logger.error(f"Error in similarity: {e}")
            logger.error(traceback.format_exc())
            return None

    def bleu_score(self, pred, ref):
        try:
            bleu = evaluate.load("bleu")
            ref_bleu = list()
            for idx, text in enumerate(ref):
                ref_bleu.append([text])
            return bleu.compute(predictions=pred, references=ref_bleu, max_order=4)['bleu']
        except Exception as e:
            logger.error(f"Error in bleu_score: {e}")
            logger.error(traceback.format_exc())
            return None

    def calculate_mgpt_fluency(self, sentence, tokenizer, model):
        tokenize_input = tokenizer.encode(sentence)
        tensor_input = torch.tensor([tokenize_input]).to(self.device)

        with torch.no_grad():
            loss = model(tensor_input, labels=tensor_input)[0]
            ppl = np.exp(loss.detach().cpu().numpy())
            return ppl

    def mgpt_fluency(self, src_sentences, trg_sentences, pred_sentences):
        try:
            model_name = 'ai-forever/mGPT'  # 'sberbank-ai/mGPT'
            tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            model = GPT2LMHeadModel.from_pretrained(model_name)
            model.to(self.device)
            model.eval()

            src_ppl = []
            trg_ppl = []
            pred_ppl = []
            src_num_skipped_sentences = 0
            trg_num_skipped_sentences = 0
            pred_num_skipped_sentences = 0

            for idx, (src_sent, trg_sent, pred_sent) in enumerate(zip(src_sentences, trg_sentences, pred_sentences)):
                try:
                    src_ppl.append(self.calculate_mgpt_fluency(src_sent, tokenizer, model))
                except Exception as e:
                    logger.error(f"Failed in mgpt fluency calculation for src sentence. Index: {idx}, src: {src_sent}, trg: {trg_sent}, pred: {pred_sent}, Error: {e}")
                    src_num_skipped_sentences += 1

                try:
                    trg_ppl.append(self.calculate_mgpt_fluency(trg_sent, tokenizer, model))
                except Exception as e:
                    logger.error(f"Failed in mgpt fluency calculation for trg sentence. Index: {idx}, src: {src_sent}, trg: {trg_sent}, pred: {pred_sent}, Error: {e}")
                    trg_num_skipped_sentences += 1

                try:
                    pred_ppl.append(self.calculate_mgpt_fluency(pred_sent, tokenizer, model))
                except Exception as e:
                    logger.error(f"Failed in mgpt fluency calculation for pred sentence. Index: {idx}, src: {src_sent}, trg: {trg_sent}, pred: {pred_sent}, Error: {e}")
                    pred_num_skipped_sentences += 1

            if src_num_skipped_sentences > 0:
                logger.error(f"{src_num_skipped_sentences} src sentences failed in fluency calculation.")
            if trg_num_skipped_sentences > 0:
                logger.error(f"{trg_num_skipped_sentences} trg sentences failed in fluency calculation.")
            if pred_num_skipped_sentences > 0:
                logger.error(f"{pred_num_skipped_sentences} pred sentences failed in fluency calculation.")

            src_fluency = sum(src_ppl) / len(src_ppl) if src_ppl else None
            trg_fluency = sum(trg_ppl) / len(trg_ppl) if trg_ppl else None
            pred_fluency = sum(pred_ppl) / len(pred_ppl) if pred_ppl else None

            return {'src_fluency': src_fluency, 'trg_fluency': trg_fluency, 'pred_fluency': pred_fluency}
        except Exception as e:
            logger.error(f"Unexpected error in mgpt fluency: {e}")
            return {'src_fluency': None, 'trg_fluency': None, 'pred_fluency': None}

    def calculate_xlmr_fluency(self, sentence, tokenizer, model):
        tensor_input = tokenizer.encode(sentence, return_tensors='pt').to(self.device)
        repeat_input = tensor_input.repeat(tensor_input.size(-1) - 2, 1)
        mask = torch.ones(tensor_input.size(-1) - 1).diag(1)[:-2].to(self.device)
        masked_input = repeat_input.masked_fill(mask == 1, tokenizer.mask_token_id)
        labels = repeat_input.masked_fill(masked_input != tokenizer.mask_token_id, -100)
        with torch.inference_mode():
            loss = model(masked_input, labels=labels).loss
            return np.exp(loss.item())

    def xlmr_fluency(self, src_sentences, trg_sentences, pred_sentences):
        try:
            model_name = "xlm-roberta-base"
            model = AutoModelForMaskedLM.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model.to(self.device)

            src_ppl = []
            trg_ppl = []
            pred_ppl = []
            src_num_skipped_sentences = 0
            trg_num_skipped_sentences = 0
            pred_num_skipped_sentences = 0

            for idx, (src_sent, trg_sent, pred_sent) in enumerate(zip(src_sentences, trg_sentences, pred_sentences)):
                try:
                    src_ppl.append(self.calculate_xlmr_fluency(src_sent, tokenizer, model))
                except Exception as e:
                    logger.error(f"Failed in xlmr fluency calculation for src sentence. Index: {idx}, src: {src_sent}, trg: {trg_sent}, pred: {pred_sent}, Error: {e}")
                    src_num_skipped_sentences += 1

                try:
                    trg_ppl.append(self.calculate_xlmr_fluency(trg_sent, tokenizer, model))
                except Exception as e:
                    logger.error(f"Failed in xlmr fluency calculation for trg sentence. Index: {idx}, src: {src_sent}, trg: {trg_sent}, pred: {pred_sent}, Error: {e}")
                    trg_num_skipped_sentences += 1

                try:
                    pred_ppl.append(self.calculate_xlmr_fluency(pred_sent, tokenizer, model))
                except Exception as e:
                    logger.error(f"Failed in xlmr fluency calculation for pred sentence. Index: {idx}, src: {src_sent}, trg: {trg_sent}, pred: {pred_sent}, Error: {e}")
                    pred_num_skipped_sentences += 1

            if src_num_skipped_sentences > 0:
                logger.error(f"{src_num_skipped_sentences} src sentences failed in fluency calculation.")
            if trg_num_skipped_sentences > 0:
                logger.error(f"{trg_num_skipped_sentences} trg sentences failed in fluency calculation.")
            if pred_num_skipped_sentences > 0:
                logger.error(f"{pred_num_skipped_sentences} pred sentences failed in fluency calculation.")

            src_fluency = sum(src_ppl) / len(src_ppl) if src_ppl else None
            trg_fluency = sum(trg_ppl) / len(trg_ppl) if trg_ppl else None
            pred_fluency = sum(pred_ppl) / len(pred_ppl) if pred_ppl else None

            return {'src_fluency': src_fluency, 'trg_fluency': trg_fluency, 'pred_fluency': pred_fluency}
        except Exception as e:
            logger.error(f"Unexpected error in xlmr fluency: {e}")
            return {'src_fluency': None, 'trg_fluency': None, 'pred_fluency': None}

    def evaluate(self):
        try:
            output_df = pd.read_csv(self.output_file)

            # output_df.fillna('', inplace=True)
            output_df['pred'].fillna(output_df['src'], inplace=True)

            output_df.loc[output_df['pred'].str.isspace(), 'pred'] = output_df['src']
            
            if self.task == 'pos_to_neg':
                target_label = 0
            else:
                target_label = 1

            accuracy = self.sentiment_accuracy(output_df['pred'].to_list(), [target_label] * len(output_df), self.lang)
            similarity_score = self.similarity(output_df['pred'].to_list(), output_df['trg'].to_list(), self.lang)
            bleu = self.bleu_score(output_df['pred'].to_list(), output_df['trg'].to_list())
            fluency_scores_mgpt = self.mgpt_fluency(output_df['src'].to_list(), output_df['trg'].to_list(), output_df['pred'].to_list())
            fluency_scores_xlmr = self.xlmr_fluency(output_df['src'].to_list(), output_df['trg'].to_list(), output_df['pred'].to_list())

            return accuracy, similarity_score, bleu, fluency_scores_mgpt, fluency_scores_xlmr
        except Exception as e:
            logger.error(f"Error in evaluate: {e}")
            return None, None, None, None, None
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TST Evaluator")
    parser.add_argument("--task", type=str, required=True, help="Task (pos_to_neg or neg_to_pos)")
    parser.add_argument("--lang", type=str, required=True, help="Language")
    parser.add_argument("--output_file", type=str, required=True, help="Output file")
    parser.add_argument("--methodology", type=str, required=True, help="Methodology")
    args = parser.parse_args()

    logger.info("Command-line arguments:")
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device {device}')    

    if torch.cuda.is_available():
        logger.info(f'GPU machine name: {torch.cuda.get_device_name(0)}')
        logger.info(f'SSH machine name: {socket.gethostname()}')
    
    tst_evaluator = TSTEvaluator(args.task, args.lang, args.output_file, args.methodology)
    tst_evaluator.set_seed(53)
    accuracy, similarity_score, bleu, fluency_scores_mgpt, fluency_scores_xlmr = tst_evaluator.evaluate()

    accuracy_text = f", Sentiment Accuracy: {accuracy:.2f}" if accuracy is not None else ", Sentiment Accuracy: None"
    similarity_score_text = f", Similarity: {similarity_score:.2f}" if similarity_score is not None else ", Similarity: None"
    bleu_text = f", Bleu Score: {bleu:.2f}" if bleu is not None else ", Bleu Score: None"

    fluency_mgpt_text = ", ".join([f"mGPT {key}: {value:.2f}" if value is not None else f"mGPT {key}: None" for key, value in fluency_scores_mgpt.items()])
    fluency_xlmr_text = ", ".join([f"XLM-R {key}: {value:.2f}" if value is not None else f"XLM-R {key}: None" for key, value in fluency_scores_xlmr.items()])
    
    logger.info(f"Language: {args.lang}, methodology: {args.methodology}, task: {args.task} {accuracy_text}{similarity_score_text}{bleu_text},{fluency_mgpt_text}, {fluency_xlmr_text}")

    logger.info('=' * 50 + '\n')