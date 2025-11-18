# # tweet_to_reply_sentiment_bert.py
# import os
# import re
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# import torch
# from torch.utils.data import Dataset
# from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
# import warnings
# from transformers import TrainingArguments, Trainer

# warnings.filterwarnings("ignore")

# # -------------------------
# # === Config / Filepaths ===
# # -------------------------
# TWEETS_CSV = r"C:\Users\kaupk\Downloads\main_tweets.csv"       # original tweets
# REPLIES_CSV = r"C:\Users\kaupk\Downloads\all_replies.csv"      # must contain tweet_id, reply_text
# SAVE_MODELS_DIR = "bert_models_output"
# os.makedirs(SAVE_MODELS_DIR, exist_ok=True)

# # -------------------------
# # === Helpers / Cleaning ===
# # -------------------------
# def clean_text(text):
#     text = "" if pd.isna(text) else str(text)
#     text = re.sub(r"http\S+", "", text)
#     text = re.sub(r"@\w+", "", text)
#     text = re.sub(r"[^\w\s]", "", text)
#     return text.lower().strip()

# # -------------------------
# # === Load CSVs ===
# # -------------------------
# tweets_df = pd.read_csv(TWEETS_CSV, header=None, names=["tweet_id", "tweet_text"])
# replies_df = pd.read_csv(REPLIES_CSV, header=None, names=["tweet_id", "reply_text"])

# # Assertions
# assert 'tweet_id' in tweets_df.columns, "tweets CSV must contain tweet_id column"
# assert 'tweet_id' in replies_df.columns, "replies CSV must contain tweet_id column"
# assert 'tweet_text' in tweets_df.columns, "tweets CSV must contain a text column (tweet_text)"
# assert 'reply_text' in replies_df.columns, "replies CSV must contain a text column (reply_text)"

# # -------------------------
# # === Clean text ===
# # -------------------------
# tweets_df['tweet_text_clean'] = tweets_df['tweet_text'].apply(clean_text)
# replies_df['reply_text_clean'] = replies_df['reply_text'].apply(clean_text)

# # -------------------------
# # === Merge replies with tweets ===
# # -------------------------
# merged = pd.merge(
#     replies_df[['tweet_id', 'reply_text_clean']],
#     tweets_df[['tweet_id', 'tweet_text_clean']],
#     on='tweet_id',
#     how='inner'
# )
# print(f"Reply-level rows: {len(merged)}, unique tweets: {merged['tweet_id'].nunique()}")

# # -------------------------
# # === Label encoding replies ===
# # -------------------------
# # For simplicity, we generate labels using a simple heuristic (VADER)
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# analyzer = SentimentIntensityAnalyzer()
# def get_compound(text):
#     return analyzer.polarity_scores(str(text))['compound']
# def vader_label_from_compound(c):
#     if c > 0.05:
#         return "positive"
#     elif c < -0.05:
#         return "negative"
#     else:
#         return "neutral"

# merged['reply_compound'] = merged['reply_text_clean'].apply(get_compound)
# merged['reply_sentiment'] = merged['reply_compound'].apply(vader_label_from_compound)

# # -------------------------
# # === Label encoding for BERT ===
# # -------------------------
# le = LabelEncoder()
# y_enc = le.fit_transform(merged['reply_sentiment'].values)
# num_classes = len(le.classes_)
# print("Class mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

# # -------------------------
# # === Prepare BERT dataset ===
# # -------------------------
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# # Combine tweet + reply as input for sequence-pair classification
# inputs = merged['tweet_text_clean'] + " [SEP] " + merged['reply_text_clean']

# encodings = tokenizer(list(inputs), truncation=True, padding=True, max_length=128)

# class TweetReplyDataset(Dataset):
#     def __init__(self, encodings, labels):
#         self.encodings = encodings
#         self.labels = labels

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, idx):
#         item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
#         item['labels'] = torch.tensor(self.labels[idx])
#         return item

# dataset = TweetReplyDataset(encodings, y_enc)

# # -------------------------
# # === Train/Test split ===
# # -------------------------
# train_size = int(0.8 * len(dataset))
# test_size = len(dataset) - train_size
# train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# # -------------------------
# # === Load BERT for classification ===
# # -------------------------
# model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_classes)

# # -------------------------
# # === Training arguments ===
# # -------------------------
# training_args = TrainingArguments(
#     output_dir=SAVE_MODELS_DIR,
#     num_train_epochs=6,
#     per_device_train_batch_size=16,
#     per_device_eval_batch_size=16,
#     warmup_steps=100,
#     weight_decay=0.01,
#     logging_dir='./logs',
#     logging_steps=50,
#     evaluation_strategy="steps",   # must be a string 'steps' or 'epoch'
#     save_strategy="steps",
#     save_steps=200,
#     load_best_model_at_end=True,
#     learning_rate=2e-5,
# )


# # -------------------------
# # === Metrics ===
# # -------------------------
# from sklearn.metrics import precision_recall_fscore_support

# def compute_metrics(pred):
#     labels = pred.label_ids
#     preds = np.argmax(pred.predictions, axis=1)
#     precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
#     acc = accuracy_score(labels, preds)
#     return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

# # -------------------------
# # === Trainer ===
# # -------------------------
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=test_dataset,
#     compute_metrics=compute_metrics
# )

# # -------------------------
# # === Train ===
# # -------------------------
# trainer.train()

# # -------------------------
# # === Evaluate ===
# # -------------------------
# results = trainer.evaluate()
# print("BERT Test Metrics:", results)

# # Predictions
# preds_output = trainer.predict(test_dataset)
# y_true = [dataset[i]['labels'].item() for i in range(len(dataset))][-test_size:]
# y_pred = np.argmax(preds_output.predictions, axis=1)

# print("Classification report:")
# print(classification_report(y_true, y_pred, target_names=le.classes_, zero_division=0))
# print("Confusion matrix:")
# print(confusion_matrix(y_true, y_pred))

# # -------------------------
# # === Save model ===
# # -------------------------
# model.save_pretrained(SAVE_MODELS_DIR)
# tokenizer.save_pretrained(SAVE_MODELS_DIR)
# print(f"\nSaved BERT model and tokenizer to {SAVE_MODELS_DIR}")

# bert_reply_sentiment.py
import os
import re
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import torch

from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments

# -------------------------
# === Config / Filepaths ===
# -------------------------
TWEETS_CSV = r"C:\Users\kaupk\Downloads\main_tweets.csv"
REPLIES_CSV = r"C:\Users\kaupk\Downloads\all_replies.csv"
OUTPUT_DIR = "bert_models_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 4
LEARNING_RATE = 2e-5

# -------------------------
# === Helpers / Cleaning ===
# -------------------------
def clean_text(text):
    text = "" if pd.isna(text) else str(text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.lower().strip()

analyzer = SentimentIntensityAnalyzer()
def get_compound(text):
    return analyzer.polarity_scores(str(text))['compound']

def vader_label_from_compound(compound):
    if compound > 0.05:
        return "positive"
    elif compound < -0.05:
        return "negative"
    else:
        return "neutral"

# -------------------------
# === Load CSVs ===
# -------------------------
tweets_df = pd.read_csv(TWEETS_CSV, header=None, names=["tweet_id","tweet_text"])
replies_df = pd.read_csv(REPLIES_CSV, header=None, names=["tweet_id","reply_text"])

# Clean text
tweets_df['clean_text'] = tweets_df['tweet_text'].apply(clean_text)
replies_df['clean_text'] = replies_df['reply_text'].apply(clean_text)

# VADER scores
tweets_df['tweet_compound'] = tweets_df['clean_text'].apply(get_compound)
tweets_df['tweet_sentiment'] = tweets_df['tweet_compound'].apply(vader_label_from_compound)
replies_df['reply_compound'] = replies_df['clean_text'].apply(get_compound)
replies_df['reply_sentiment'] = replies_df['reply_compound'].apply(vader_label_from_compound)

# -------------------------
# === Merge replies with tweets ===
# -------------------------
merged = pd.merge(
    replies_df[['tweet_id','clean_text','reply_sentiment']],
    tweets_df[['tweet_id','clean_text','tweet_compound']],
    on='tweet_id', how='inner',
    suffixes=('_reply','_tweet')
)
merged = merged.rename(columns={'clean_text_reply':'reply_text','clean_text_tweet':'tweet_text'})

print(f"Total replies: {len(merged)}, Unique tweets: {merged['tweet_id'].nunique()}")

# -------------------------
# === Encode labels ===
# -------------------------
le = LabelEncoder()
merged['reply_sentiment_enc'] = le.fit_transform(merged['reply_sentiment'])
num_classes = len(le.classes_)
print("Class mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

# Compute class weights
class_weights = compute_class_weight(class_weight='balanced',
                                     classes=np.unique(merged['reply_sentiment_enc']),
                                     y=merged['reply_sentiment_enc'])
class_weights = torch.tensor(class_weights,dtype=torch.float)

# -------------------------
# === Train/test split ===
# -------------------------
X = merged['reply_text'].tolist()
y = merged['reply_sentiment_enc'].tolist()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------
# === Tokenization ===
# -------------------------
tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=MAX_LENGTH)
test_encodings = tokenizer(X_test, truncation=True, padding=True, max_length=MAX_LENGTH)

class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

train_dataset = SentimentDataset(train_encodings, y_train)
test_dataset = SentimentDataset(test_encodings, y_test)

# -------------------------
# === Model setup ===
# -------------------------
model = BertForSequenceClassification.from_pretrained(MODEL_NAME,
                                                      num_labels=num_classes)

# -------------------------
# === Training ===
# -------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    eval_strategy="steps",  # or "epoch"
    save_strategy="steps",
    save_total_limit=2,
    logging_dir='./logs',
    logging_steps=50,
    learning_rate=LEARNING_RATE,
    load_best_model_at_end=True
)


# Metrics for Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

trainer.train()

# -------------------------
# === Evaluation ===
# -------------------------
# Evaluate manually
eval_results = trainer.evaluate()
print("Evaluation results:", eval_results)

# Save model
trainer.save_model(OUTPUT_DIR)
print("\nBERT Evaluation Results:")
print(eval_results)

# Predictions
preds_output = trainer.predict(test_dataset)
pred_labels = np.argmax(preds_output.predictions, axis=1)
print("\nClassification Report:")
print(classification_report(y_test, pred_labels, target_names=le.classes_))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, pred_labels))
