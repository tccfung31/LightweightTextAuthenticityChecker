import json
import matplotlib.pyplot as plt
import numpy as np
from rapidfuzz import fuzz
from sklearn.metrics import accuracy_score, f1_score, brier_score_loss
import xgboost as xgb
from sklearn.model_selection import train_test_split
import time
import pickle
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GPT2Tokenizer, GPT2LMHeadModel
import torch
import logging
import tkinter as tk
from tkinter import filedialog
from nltk.tokenize import sent_tokenize
from pdfminer.high_level import extract_text
import nltk
import shap

# --- Configuration & Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK 'punkt' tokenizer...")
    nltk.download('punkt')

# Global Feature Calculation Parameters
NGRAM_NUM = 5
CUTOFF_START = 4
CUTOFF_END = 800
FIXED_FEATURE_SIZE = 2 * NGRAM_NUM + 3

RESERVED_KEYS = {'input', 'common_features', 'avg_common_features',
                 'common_features_ori_vs_allcombined', 'fzwz_features', 'diff'}

T5_PROMPTS = [
    "Help me polish this",
    "Rewrite this for me",
    "Make this fluent while doing minimal change",
    "Refine this for me please, just show me the out put",
    "Concise this for me and keep all the information"
]
NUM_T5_REWRITES = len(T5_PROMPTS)

def _load_gpt2():
    try:
        logging.info("Loading GPT-2 model for perplexity (optional)...")
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        logging.info("GPT-2 model loaded.")
        return tokenizer, model
    except Exception as e:
        logging.warning(f"Could not load GPT-2 for perplexity calculation. Error: {e}")
        return None, None

def calculate_perplexity(text, tokenizer, model):
    if not tokenizer or not model:
        logging.warning("Tokenizer/model not available for perplexity calculation.")
        return 0.0
    if not text or not isinstance(text, str):
        return 0.0
    try:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=1024)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs['input_ids'])
        loss = outputs.loss
        if loss is not None and loss.numel() == 1:
            return torch.exp(loss).item()
        else:
            return 0.0
    except Exception as e:
        logging.error(f"Error calculating perplexity: {e}")
        return 0.0

def tokenize_and_normalize(sentence):
    if not isinstance(sentence, str):
        return []
    return [word.lower().strip() for word in sentence.split() if word.strip()]

def extract_ngrams(tokens, n):
    if len(tokens) < n:
        return []
    return [' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

def calculate_common_ngrams(tokens1, tokens2, max_n):
    common_counts = [0] * max_n
    for n in range(1, max_n + 1):
        ngrams1 = set(extract_ngrams(tokens1, n))
        if not ngrams1:
            continue
        ngrams2 = set(extract_ngrams(tokens2, n))
        common_counts[n-1] = len(ngrams1.intersection(ngrams2))
    return common_counts

def process_single_record(record):
    original_text = record.get('input', '')
    if not isinstance(original_text, str):
        original_text = str(original_text)
    original_tokens = tokenize_and_normalize(original_text)
    original_len = len(original_tokens)
    if original_len < CUTOFF_START or original_len > CUTOFF_END:
        return None
    rewrite_keys = [k for k in record if k not in RESERVED_KEYS]
    num_valid_rewrites = 0
    total_rewrite_len = 0
    all_common_ngram_counts = [0] * NGRAM_NUM
    fuzz_ratios = []
    token_set_ratios = []
    combined_rewrites_text = ''
    for key in rewrite_keys:
        rewrite_text = record.get(key, '')
        if not isinstance(rewrite_text, str):
            rewrite_text = str(rewrite_text)
        if rewrite_text:
            rewrite_tokens = tokenize_and_normalize(rewrite_text)
            combined_rewrites_text += (' ' + rewrite_text)
            num_valid_rewrites += 1
            total_rewrite_len += len(rewrite_tokens)
            common_counts = calculate_common_ngrams(original_tokens, rewrite_tokens, NGRAM_NUM)
            for i in range(NGRAM_NUM):
                all_common_ngram_counts[i] += common_counts[i]
            try:
                fuzz_ratios.append(fuzz.ratio(original_text, rewrite_text))
                token_set_ratios.append(fuzz.token_set_ratio(original_text, rewrite_text))
            except Exception as e:
                logging.warning(f"Fuzz ratio calculation failed for key '{key}'. Error: {e}")
                fuzz_ratios.append(0.0)
                token_set_ratios.append(0.0)
    processed = {'input': original_text, 'original_len': original_len}
    if num_valid_rewrites > 0:
        processed['avg_common_features'] = [count / num_valid_rewrites for count in all_common_ngram_counts]
        processed['avg_len_diff'] = (total_rewrite_len / num_valid_rewrites) - original_len
        processed['avg_fuzz_ratio'] = np.mean(fuzz_ratios) if fuzz_ratios else 0.0
        processed['avg_token_set_ratio'] = np.mean(token_set_ratios) if token_set_ratios else 0.0
    else:
        processed['avg_common_features'] = [0.0] * NGRAM_NUM
        processed['avg_len_diff'] = -original_len
        processed['avg_fuzz_ratio'] = 0.0
        processed['avg_token_set_ratio'] = 0.0
    combined_tokens = tokenize_and_normalize(combined_rewrites_text.strip())
    processed['common_ori_vs_combined'] = calculate_common_ngrams(original_tokens, combined_tokens, NGRAM_NUM)
    return processed

def extract_feature_vector(processed_record, fixed_size=FIXED_FEATURE_SIZE):
    if processed_record is None:
        return None
    original_len = processed_record.get('original_len', 0)
    norm_factor = float(max(1, original_len))
    features = []
    avg_common = processed_record.get('avg_common_features', [0.0] * NGRAM_NUM)
    features.extend([val / norm_factor for val in avg_common[:NGRAM_NUM]])
    ori_vs_combined = processed_record.get('common_ori_vs_combined', [0.0] * NGRAM_NUM)
    features.extend([val / norm_factor for val in ori_vs_combined[:NGRAM_NUM]])
    features.append(processed_record.get('avg_fuzz_ratio', 0.0) / 100.0)
    features.append(processed_record.get('avg_token_set_ratio', 0.0) / 100.0)
    features.append(processed_record.get('avg_len_diff', 0.0) / norm_factor)
    current_len = len(features)
    if current_len < fixed_size:
        features.extend([0.0] * (fixed_size - current_len))
    elif current_len > fixed_size:
        logging.warning(f"Feature vector size ({current_len}) > fixed size ({fixed_size}). Truncating.")
        features = features[:fixed_size]
    if len(features) != fixed_size:
        logging.error(f"Final feature vector size mismatch: {len(features)} != {fixed_size}. Returning None.")
        return None
    return np.array(features)

def prepare_features_for_dataset(raw_data_list):
    all_features = []
    original_lengths = []
    for i, record in enumerate(raw_data_list):
        if (i + 1) % 500 == 0:
            logging.info(f"Processing record {i+1}/{len(raw_data_list)} for feature extraction...")
        processed = process_single_record(record)
        if processed:
            feature_vector = extract_feature_vector(processed)
            if feature_vector is not None:
                all_features.append(feature_vector)
                original_lengths.append(processed['original_len'])
    if not all_features:
        logging.warning("No features extracted. Returning empty arrays.")
        return np.empty((0, FIXED_FEATURE_SIZE)), np.array([])
    return np.vstack(all_features), np.array(original_lengths)

def train_evaluate_xgboost(gpt_features, human_features, gpt_lengths, human_lengths):
    if gpt_features.shape[0] == 0 or human_features.shape[0] == 0:
        logging.error("Cannot train model: One or both feature sets are empty.")
        return None
    logging.info(f"Training data shapes: GPT={gpt_features.shape}, Human={human_features.shape}")
    X = np.concatenate((gpt_features, human_features), axis=0)
    y = np.concatenate((np.ones(gpt_features.shape[0]), np.zeros(human_features.shape[0])), axis=0)
    lengths = np.concatenate((gpt_lengths, human_lengths), axis=0)
    min_samples_per_class = 10
    test_size = 0.2
    if gpt_features.shape[0] < min_samples_per_class or human_features.shape[0] < min_samples_per_class or (gpt_features.shape[0] + human_features.shape[0]) * test_size < 2:
        logging.warning("Not enough samples for train/test split. Training on all data.")
        X_train, y_train = X, y
        X_test, y_test, len_test = None, None, None
    else:
        try:
            X_train, X_test, y_train, y_test, len_train, len_test = train_test_split(
                X, y, lengths, test_size=test_size, random_state=42, stratify=y
            )
            logging.info(f"Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
        except ValueError:
            logging.warning("Stratified split failed. Falling back to non-stratified.")
            X_train, X_test, y_train, y_test, len_train, len_test = train_test_split(
                X, y, lengths, test_size=test_size, random_state=42
            )
            logging.info(f"Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
    logging.info("Training XGBoost classifier...")
    clf = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42
    )
    clf.fit(X_train, y_train)
    logging.info("Training complete.")
    if X_test is not None and y_test is not None and len(y_test) > 0:
        logging.info("Evaluating classifier on test data...")
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0.0)
        brier = brier_score_loss(y_test, y_prob[:, 1])
        confidence = np.mean(np.max(y_prob, axis=1))
        logging.info("--- Overall Test Metrics ---")
        logging.info(f" Accuracy: {accuracy:.4f}")
        logging.info(f" F1 Score: {f1:.4f}")
        logging.info(f" Brier Score (MSE): {brier:.4f}")
        logging.info(f" Average Confidence: {confidence:.4f}")
        logging.info("----------------------------")
        plot_length_ablation(y_test, y_pred, len_test, "len_ablation_f1.png")
    return clf

def plot_length_ablation(y_true, y_pred, lengths, filename="length_ablation.png"):
    logging.info("Performing length ablation study...")
    length_bins = [(0, 25), (25, 50), (50, 75), (75, 100), (100, 150), (150, 300), (300, CUTOFF_END)]
    f1_scores = []
    bin_labels = []
    bin_counts = []
    for start, finish in length_bins:
        mask = (lengths > start) & (lengths <= finish)
        if np.sum(mask) > 0:
            sub_y_true = y_true[mask]
            sub_y_pred = y_pred[mask]
            sub_f1 = f1_score(sub_y_true, sub_y_pred, zero_division=0.0)
            count = len(sub_y_true)
            label = f"{start+1}-{finish}"
            f1_scores.append(sub_f1)
            bin_labels.append(label)
            bin_counts.append(count)
            logging.info(f" Length {label}: F1={sub_f1:.4f}, Samples={count}")
        else:
            logging.info(f" Length {start+1}-{finish}: No samples in this bin.")
    if bin_labels and f1_scores:
        plt.figure(figsize=(10, 6))
        plt.plot(bin_labels, f1_scores, '-o', color='blue', label='Detection F1 Score by Length')
        for i, count in enumerate(bin_counts):
            if count > 0:
                plt.text(bin_labels[i], f1_scores[i] + 0.01, str(count), fontsize=9, ha='center', va='bottom')
        plt.title('Model F1 Score vs. Text Length (Test Set)')
        plt.xlabel('Text Length Bins (Tokens)')
        plt.ylabel('F1 Score')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1.05)
        plt.legend()
        plt.grid(True, axis='y', linestyle='--')
        plt.tight_layout()
        try:
            plt.savefig(filename)
            logging.info(f"Saved length ablation plot to {filename}")
        except Exception as e:
            logging.warning(f"Could not save plot '{filename}'. Error: {e}")
    else:
        logging.info("No data available for length ablation plot.")

def load_t5_model(model_name='t5-small'):
    try:
        logging.info(f"Loading T5 model ({model_name}) and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        logging.info(f"T5 model '{model_name}' loaded successfully on {device}.")
        return tokenizer, model, device
    except Exception as e:
        logging.error(f"Error loading T5 model/tokenizer '{model_name}': {e}")
        return None, None, None

def generate_rewrites_with_prompts(text, tokenizer, model, device, prompts=T5_PROMPTS, temperature=1.5, num_beams=5):
    if not tokenizer or not model:
        logging.error("T5 Tokenizer or Model not available for generating rewrites.")
        return {}
    rewritten_dict = {}
    for i, prompt in enumerate(prompts):
        input_text = f"{prompt}: {text}"
        try:
            inputs = tokenizer(input_text, return_tensors='pt', max_length=512, truncation=True, padding=True).to(device)
            input_length = inputs['input_ids'].shape[1]
            max_len = min(int(input_length * 1.5) + 20, 512)
            min_len = max(10, input_length // 2)
            if min_len >= max_len:
                min_len = max(1, max_len // 2)
            outputs = model.generate(
                **inputs,
                num_beams=num_beams,
                do_sample=True,
                temperature=temperature,
                top_k=50,
                top_p=0.95,
                num_return_sequences=1,
                max_length=max_len,
                min_length=min_len,
                early_stopping=True,
                repetition_penalty=1.2
            )
            rewrite = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            rewrite_key = f'rewrite_{i+1}'
            rewritten_dict[rewrite_key] = rewrite
        except Exception as e:
            logging.error(f"Error generating rewrite for prompt '{prompt}': {e}")
            rewritten_dict[f'rewrite_{i+1}'] = ""
    return rewritten_dict

def classify_text(text, clf, t5_tokenizer, t5_model, t5_device, explainer=None):
    if not text or not isinstance(text, str):
        logging.error("Invalid input text provided.")
        return None, None, None
    if not clf or not t5_tokenizer or not t5_model:
        logging.error("Classifier or T5 model/tokenizer not available.")
        return None, None, None
    logging.info("Generating rewrites for input...")
    start_time = time.time()
    rewrites = generate_rewrites_with_prompts(text, t5_tokenizer, t5_model, t5_device)
    logging.info(f"Generated {len([r for r in rewrites.values() if r])} non-empty rewrites in {time.time() - start_time:.2f}s.")
    data_record = {'input': text}
    data_record.update(rewrites)
    processed = process_single_record(data_record)
    if not processed:
        logging.warning("Input text filtered out (e.g., length). Cannot classify.")
        return None, None, None
    feature_vector = extract_feature_vector(processed)
    if feature_vector is None:
        logging.error("Feature extraction failed for the input text.")
        return None, None, None
    try:
        prob = clf.predict_proba(feature_vector.reshape(1, -1))[0]
        prob_gpt = prob[1]
        prediction = "AI-Generated" if prob_gpt > 0.5 else "Human-Written"
        confidence = prob_gpt if prediction == "AI-Generated" else prob[0]
        shap_values = None
        if explainer:
            shap_values = explainer.shap_values(feature_vector.reshape(1, -1))
        return prediction, confidence, shap_values
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return None, None, None

def examine_external_dataset(file_path, clf, true_label, t5_tokenizer, t5_model, t5_device):
    logging.info(f"--- Examining External Dataset: {file_path} ---")
    if true_label not in [0, 1]:
        logging.error("true_label must be 0 (Human) or 1 (AI).")
        return
    if not clf or not t5_tokenizer or not t5_model:
        logging.error("Classifier or T5 components missing for external evaluation.")
        return
    try:
        with open(file_path, "r", encoding='utf-8') as f:
            external_data_raw = json.load(f)
        logging.info(f"Loaded {len(external_data_raw)} raw records from {file_path}.")
    except Exception as e:
        logging.error(f"Failed to load or parse {file_path}: {e}")
        return
    if not external_data_raw:
        logging.error("External dataset file is empty.")
        return
    all_features = []
    valid_lengths = []
    true_labels = []
    processed_count = 0
    skipped_count = 0
    start_time_all = time.time()
    for i, item in enumerate(external_data_raw):
        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time_all
            logging.info(f" Processing external record {i+1}/{len(external_data_raw)}... (skipped {skipped_count}, elapsed {elapsed:.1f}s)")
        input_text = item.get('input', '')
        if not isinstance(input_text, str):
            input_text = str(input_text)
        if not input_text:
            skipped_count += 1
            continue
        rewrites = generate_rewrites_with_prompts(input_text, t5_tokenizer, t5_model, t5_device)
        data_record = {'input': input_text}
        data_record.update(rewrites)
        processed = process_single_record(data_record)
        if processed:
            feature_vector = extract_feature_vector(processed)
            if feature_vector is not None:
                all_features.append(feature_vector)
                valid_lengths.append(processed['original_len'])
                true_labels.append(true_label)
                processed_count += 1
            else:
                skipped_count += 1
        else:
            skipped_count += 1
    total_time = time.time() - start_time_all
    logging.info(f"Finished processing external dataset in {total_time:.2f}s. Processed: {processed_count}, Skipped: {skipped_count}")
    if not all_features:
        logging.error("No valid features extracted from the external dataset after processing/filtering.")
        return
    X_external = np.vstack(all_features)
    y_true_external = np.array(true_labels)
    len_external = np.array(valid_lengths)
    logging.info(f"Extracted features for {X_external.shape[0]} samples.")
    logging.info("Predicting labels for external dataset...")
    y_pred_external = clf.predict(X_external)
    y_prob_external = clf.predict_proba(X_external)
    accuracy = accuracy_score(y_true_external, y_pred_external)
    f1 = f1_score(y_true_external, y_pred_external, pos_label=true_label, zero_division=0.0)
    brier_prob_for_true_class = y_prob_external[:, true_label] if true_label == 1 else y_prob_external[:, 0]
    brier = brier_score_loss(y_true_external, brier_prob_for_true_class)
    confidence = np.mean(np.max(y_prob_external, axis=1))
    logging.info("--- External Dataset Evaluation Results ---")
    logging.info(f" Dataset Path: {file_path}")
    logging.info(f" Assumed True Label: {'AI (1)' if true_label == 1 else 'Human (0)'}")
    logging.info(f" Samples Evaluated: {X_external.shape[0]}")
    logging.info(f" Accuracy: {accuracy:.4f}")
    logging.info(f" F1 Score (for class {true_label}): {f1:.4f}")
    logging.info(f" Brier Score (MSE for true class {true_label}): {brier:.4f}")
    logging.info(f" Average Confidence (max prob): {confidence:.4f}")
    logging.info("-----------------------------------------")
    plot_length_ablation(y_true_external, y_pred_external, len_external, f"external_{os.path.basename(file_path)}_ablation.png")

def load_data(file_path):
    try:
        with open(file_path, "r", encoding='utf-8') as f:
            data = json.load(f)
        logging.info(f"Successfully loaded {len(data)} records from {file_path}")
        return data
    except FileNotFoundError:
        logging.error(f"Data file not found: {file_path}")
        return None
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from file: {file_path}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred loading {file_path}: {e}")
        return None

def load_or_train_model(gpt_data_path, human_data_path, model_save_path, force_retrain=False):
    clf = None
    if not force_retrain and os.path.exists(model_save_path):
        try:
            with open(model_save_path, 'rb') as f:
                clf = pickle.load(f)
            logging.info(f"Loaded trained XGBoost model from {model_save_path}")
        except Exception as e:
            logging.warning(f"Error loading model from {model_save_path}: {e}. Retraining...")
            force_retrain = True
    if force_retrain or clf is None:
        logging.info("--- Starting Model Training ---")
        data_gpt = load_data(gpt_data_path)
        data_human = load_data(human_data_path)
        if data_gpt is None or data_human is None:
            logging.error("Cannot train model without both GPT and Human data. Exiting.")
            return None
        logging.info("Extracting features for GPT data...")
        gpt_features, gpt_lengths = prepare_features_for_dataset(data_gpt)
        logging.info("Extracting features for Human data...")
        human_features, human_lengths = prepare_features_for_dataset(data_human)
        if gpt_features.shape[0] > 0 and human_features.shape[0] > 0:
            clf = train_evaluate_xgboost(gpt_features, human_features, gpt_lengths, human_lengths)
            if clf:
                try:
                    with open(model_save_path, 'wb') as f:
                        pickle.dump(clf, f)
                    logging.info(f"Saved trained model to {model_save_path}")
                except Exception as e:
                    logging.error(f"Error saving model to {model_save_path}: {e}")
            else:
                logging.error("Model training failed.")
        else:
            logging.error("Feature extraction resulted in empty dataset(s). Cannot train.")
            clf = None
        logging.info("--- Finished Model Training ---")
    return clf

# --- GUI Functions ---

def import_pdf(text_widget, status_widget):
    file_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
    if file_path:
        try:
            text = extract_text(file_path)
            text_widget.delete("1.0", tk.END)
            text_widget.insert(tk.END, text)
            status_widget.config(text="PDF loaded successfully.")
        except Exception as e:
            status_widget.config(text=f"Error loading PDF: {e}")
    else:
        status_widget.config(text="No file selected.")

def scan_text(input_widget, result_widget, prediction_label, status_widget, clf, t5_tok, t5_mod, t5_dev, explainer):
    status_widget.config(text="Scanning...")
    root.update()
    full_text = input_widget.get("1.0", tk.END).strip()
    if not full_text:
        status_widget.config(text="No text to scan.")
        return
    result_widget.delete("1.0", tk.END)
    result_widget.insert(tk.END, full_text)
    result = classify_text(full_text, clf, t5_tok, t5_mod, t5_dev)
    if not result:
        status_widget.config(text="Error classifying the text.")
        return
    overall_prediction, overall_confidence, _ = result
    prediction_label.config(text=f"Overall Prediction: {overall_prediction} (Confidence: {overall_confidence*100:.2f}%)")
    sentences = sent_tokenize(full_text)
    if sentences:
        highlight_suspected_sentences(result_widget, sentences, full_text, clf, t5_tok, t5_mod, t5_dev, explainer)
    status_widget.config(text="Scan complete.")
    result_widget.config(state="disabled")

def highlight_suspected_sentences(text_widget, sentences, full_text, clf, t5_tok, t5_mod, t5_dev, explainer):
    text_widget.tag_config("suspected", background="yellow")
    start_pos = 0
    for sentence in sentences:
        try:
            start_idx = full_text.index(sentence, start_pos)
            end_idx = start_idx + len(sentence)
            start_pos = end_idx
            prediction, confidence, shap_values = classify_text(sentence, clf, t5_tok, t5_mod, t5_dev, explainer)
            if prediction == "AI-Generated" and shap_values is not None:
                # Use SHAP values to determine highlighting: highlight if sum of positive SHAP values exceeds threshold
                shap_sum = np.sum(shap_values[0]) if shap_values[0].ndim == 1 else np.sum(shap_values[0][1])  # Adjust based on XGBoost output
                if shap_sum > 0:  # Simple threshold: positive contribution to AI class
                    start_line, start_col = get_line_col(full_text, start_idx)
                    end_line, end_col = get_line_col(full_text, end_idx)
                    text_widget.tag_add(
                        "suspected",
                        f"{start_line}.{start_col}",
                        f"{end_line}.{end_col}"
                    )
        except ValueError:
            continue

def get_line_col(text, char_idx):
    lines = text.split('\n')
    current_pos = 0
    for line_num, line in enumerate(lines, 1):
        line_len = len(line) + 1
        if current_pos + line_len > char_idx:
            col = char_idx - current_pos
            return line_num, col
        current_pos += line_len
    return len(lines), len(lines[-1]) if lines else 0

def save_to_json(input_widget, t5_tok, t5_mod, t5_dev, status_widget):
    full_text = input_widget.get("1.0", tk.END).strip()
    if not full_text:
        status_widget.config(text="No text to save.")
        return
    rewrites = generate_rewrites_with_prompts(full_text, t5_tok, t5_mod, t5_dev)
    data_record = {'input': full_text}
    data_record.update(rewrites)
    file_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
    if file_path:
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data_record, f, ensure_ascii=False, indent=4)
            status_widget.config(text=f"Saved to {file_path}")
        except Exception as e:
            status_widget.config(text=f"Error saving file: {e}")
    else:
        status_widget.config(text="Save cancelled.")

# --- Main Execution ---

if __name__ == "__main__":
    main_start_time = time.time()
    GPT_TRAIN_DATA = ""
    HUMAN_TRAIN_DATA = ""
    MODEL_SAVE_PATH = 'xgboost_detector_model_simplified.pkl'
    FORCE_RETRAIN_MODEL = True
    T5_MODEL_NAME = 't5-small'
    classifier = load_or_train_model(GPT_TRAIN_DATA, HUMAN_TRAIN_DATA, MODEL_SAVE_PATH, FORCE_RETRAIN_MODEL)
    t5_tokenizer, t5_model, t5_device = None, None, None
    explainer = None
    if classifier:
        t5_tokenizer, t5_model, t5_device = load_t5_model(T5_MODEL_NAME)
        explainer = shap.TreeExplainer(classifier)
    else:
        logging.warning("Classifier not available. Skipping T5 loading and GUI.")
    if classifier and t5_tokenizer and t5_model:
        root = tk.Tk()
        root.title("AI Text Detector")
        root.geometry("1200x600")
        main_frame = tk.Frame(root)
        main_frame.pack(fill="both", expand=True)

        # Left frame for input
        left_frame = tk.Frame(main_frame)
        left_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        tk.Label(left_frame, text="Input Text:").pack(anchor="w")
        input_text_frame = tk.Frame(left_frame)
        input_text_frame.pack(fill="both", expand=True)
        input_scrollbar = tk.Scrollbar(input_text_frame)
        input_scrollbar.pack(side="right", fill="y")
        input_text = tk.Text(input_text_frame, height=20, width=50, yscrollcommand=input_scrollbar.set)
        input_text.pack(side="left", fill="both", expand=True)
        input_scrollbar.config(command=input_text.yview)
        button_frame = tk.Frame(left_frame)
        button_frame.pack(side="bottom", fill="x")
        import_button = tk.Button(button_frame, text="Import PDF", command=lambda: import_pdf(input_text, status_label))
        import_button.pack(side="left", padx=5)
        scan_button = tk.Button(button_frame, text="Scan", command=lambda: scan_text(input_text, result_text, prediction_label, status_label, classifier, t5_tokenizer, t5_model, t5_device, explainer))
        scan_button.pack(side="left", padx=5)
        save_button = tk.Button(button_frame, text="Save to JSON", command=lambda: save_to_json(input_text, t5_tokenizer, t5_model, t5_device, status_label))
        save_button.pack(side="left", padx=5)

        # Right frame for results
        right_frame = tk.Frame(main_frame)
        right_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        prediction_label = tk.Label(right_frame, text="", anchor="w")
        prediction_label.pack(fill="x")
        tk.Label(right_frame, text="Scan Results:").pack(anchor="w")
        result_text_frame = tk.Frame(right_frame)
        result_text_frame.pack(fill="both", expand=True)
        result_scrollbar = tk.Scrollbar(result_text_frame)
        result_scrollbar.pack(side="right", fill="y")
        result_text = tk.Text(result_text_frame, height=20, width=50, yscrollcommand=result_scrollbar.set)
        result_text.pack(side="left", fill="both", expand=True)
        result_scrollbar.config(command=result_text.yview)

        # Status label
        status_label = tk.Label(root, text="Ready", anchor="w", relief="sunken")
        status_label.pack(side="bottom", fill="x")
        root.mainloop()
    main_end_time = time.time()
    logging.info(f"\nTotal script runtime: {main_end_time - main_start_time:.2f} seconds")
