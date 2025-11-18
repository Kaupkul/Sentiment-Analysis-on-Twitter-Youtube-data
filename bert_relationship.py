import os
import re
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

# -------------------------
# Config / Filepaths
# -------------------------
TWEETS_CSV = r"main_tweets.csv"
REPLIES_CSV = r"all_replies.csv"
OUTPUT_DIR = "bert_reply_sentiment_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 4
LEARNING_RATE = 2e-5

# -------------------------
# Helpers / Cleaning
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
# Load CSVs
# -------------------------
tweets_df = pd.read_csv(TWEETS_CSV, header=None, names=["tweet_id","tweet_text"])
replies_df = pd.read_csv(REPLIES_CSV, header=None, names=["tweet_id","reply_text"])

# Clean text
tweets_df['clean_text'] = tweets_df['tweet_text'].apply(clean_text)
replies_df['clean_text'] = replies_df['reply_text'].apply(clean_text)

# Compute tweet sentiment
tweets_df['tweet_compound'] = tweets_df['clean_text'].apply(get_compound)
tweets_df['tweet_sentiment'] = tweets_df['tweet_compound'].apply(vader_label_from_compound)

# Compute reply sentiment
replies_df['reply_compound'] = replies_df['clean_text'].apply(get_compound)
replies_df['reply_sentiment'] = replies_df['reply_compound'].apply(vader_label_from_compound)

# -------------------------
# Merge tweet and reply info
# -------------------------
merged = pd.merge(
    replies_df[['tweet_id','clean_text','reply_sentiment']],
    tweets_df[['tweet_id','clean_text','tweet_compound']],
    on='tweet_id', how='inner',
    suffixes=('_reply','_tweet')
)
merged = merged.rename(columns={'clean_text_reply':'reply_text','clean_text_tweet':'tweet_text'})

# -------------------------
# Create input text for BERT
# -------------------------
merged['input_text'] = "Tweet: " + merged['tweet_text'] + " [SEP] Reply: " + merged['reply_text']

# -------------------------
# Encode target labels
# -------------------------
le = LabelEncoder()
merged['reply_sentiment_enc'] = le.fit_transform(merged['reply_sentiment'])
num_classes = len(le.classes_)
print("Class mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

# Compute class weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(merged['reply_sentiment_enc']),
    y=merged['reply_sentiment_enc']
)
class_weights = torch.tensor(class_weights,dtype=torch.float)

# -------------------------
# Train/test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    merged['input_text'].tolist(),
    merged['reply_sentiment_enc'].tolist(),
    test_size=0.2, random_state=42, stratify=merged['reply_sentiment_enc']
)

# -------------------------
# Tokenization
# -------------------------
tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=MAX_LENGTH)
test_encodings = tokenizer(X_test, truncation=True, padding=True, max_length=MAX_LENGTH)

# -------------------------
# Dataset Class
# -------------------------
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
# Model
# -------------------------
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_classes)

# -------------------------
# Training Arguments
# -------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    eval_strategy="steps",
    save_strategy="steps",
    save_total_limit=2,
    logging_dir='./logs',
    logging_steps=50,
    learning_rate=LEARNING_RATE,
    load_best_model_at_end=True
)

# -------------------------
# Metrics
# -------------------------
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

# -------------------------
# Trainer
# -------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

trainer.train()

# -------------------------
# Evaluation
# -------------------------
eval_results = trainer.evaluate()
print("\nBERT Evaluation Results:")
print(eval_results)

# Predictions
preds_output = trainer.predict(test_dataset)
pred_labels = np.argmax(preds_output.predictions, axis=1)
print("\nClassification Report:")
print(classification_report(y_test, pred_labels, target_names=le.classes_))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, pred_labels))

# Save model
trainer.save_model(OUTPUT_DIR)

