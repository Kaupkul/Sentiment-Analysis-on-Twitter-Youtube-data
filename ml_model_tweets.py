
import os
import re
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

from collections import Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# sparse utilities
from scipy.sparse import hstack, csr_matrix

# Optional NN imports
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    tf_available = True
except Exception:
    tf_available = False

# Optional SMOTE
try:
    from imblearn.over_sampling import SMOTE
    smote_available = True
except Exception:
    smote_available = False

# -------------------------
# === Config / Filepaths ===
# -------------------------
TWEETS_CSV = r"main_tweets.csv"  # headerless: tweet_id, tweet_text
REPLIES_CSV = r"all_replies.csv"  # headerless: tweet_id, reply_text

# TF-IDF sizes (control memory)
TFIDF_MAX_FEATURES_TWEET = 2000
TFIDF_MAX_FEATURES_REPLY = 3000

INCLUDE_TWEET_LEN = True   # include tweet length
INCLUDE_REPLY_LEN = False  # include reply length as additional numeric (optional)

SCALE_FEATURES = True

# SMOTE & class imbalance
USE_SMOTE = True
SMOTE_K_NEIGHBORS = 5

# NN training
USE_FOCAL_LOSS = True
NN_EPOCHS = 50
NN_BATCH = 64
NN_LR = 1e-4
NN_PATIENCE = 4

# Outputs
SAVE_MODELS_DIR = "models_output"
os.makedirs(SAVE_MODELS_DIR, exist_ok=True)

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
# === Load CSVs (headerless) ===
# -------------------------
print("Loading CSVs (headerless assumed)...")
tweets_df = pd.read_csv(TWEETS_CSV, header=None, names=["tweet_id", "tweet_text"])
replies_df = pd.read_csv(REPLIES_CSV, header=None, names=["tweet_id", "reply_text"])

# try to auto-detect if file actually had headers
if 'tweet_id' not in tweets_df.columns:
    for c in tweets_df.columns:
        if 'id' in str(c).lower():
            tweets_df = tweets_df.rename(columns={c: 'tweet_id'})
            break
if 'tweet_id' not in replies_df.columns:
    for c in replies_df.columns:
        if 'id' in str(c).lower():
            replies_df = replies_df.rename(columns={c: 'tweet_id'})
            break

if 'tweet_text' not in tweets_df.columns:
    for c in tweets_df.columns:
        if 'text' in str(c).lower():
            tweets_df = tweets_df.rename(columns={c: 'tweet_text'})
            break
if 'reply_text' not in replies_df.columns:
    for c in replies_df.columns:
        if 'text' in str(c).lower():
            replies_df = replies_df.rename(columns={c: 'reply_text'})
            break

assert 'tweet_id' in tweets_df.columns, "tweets CSV must contain tweet_id column"
assert 'tweet_id' in replies_df.columns, "replies CSV must contain tweet_id column"
assert 'tweet_text' in tweets_df.columns, "tweets CSV must contain a text column"
assert 'reply_text' in replies_df.columns, "replies CSV must contain a text column"

# -------------------------
# === Clean text and VADER ===
# -------------------------
print("Cleaning text and computing VADER compound scores...")
tweets_df['clean_text'] = tweets_df['tweet_text'].apply(clean_text)
replies_df['clean_text'] = replies_df['reply_text'].apply(clean_text)

tweets_df['tweet_compound'] = tweets_df['clean_text'].apply(get_compound)
replies_df['reply_compound'] = replies_df['clean_text'].apply(get_compound)

tweets_df['tweet_sentiment_label'] = tweets_df['tweet_compound'].apply(vader_label_from_compound)
replies_df['reply_sentiment_label'] = replies_df['reply_compound'].apply(vader_label_from_compound)

if INCLUDE_TWEET_LEN:
    tweets_df['tweet_len'] = tweets_df['clean_text'].apply(lambda t: len(t))
else:
    tweets_df['tweet_len'] = 0

if INCLUDE_REPLY_LEN:
    replies_df['reply_len'] = replies_df['clean_text'].apply(lambda t: len(t))
else:
    replies_df['reply_len'] = 0

# -------------------------
# === Build TF-IDF for tweets and replies ===
# -------------------------
print("Fitting TF-IDF for tweets and replies...")
tweet_texts = tweets_df['clean_text'].fillna("").tolist()
reply_texts = replies_df['clean_text'].fillna("").tolist()

tfidf_tweet = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES_TWEET, ngram_range=(1,2))
X_tfidf_tweet = tfidf_tweet.fit_transform(tweet_texts)  # shape (n_tweets, ft_tweet)
joblib.dump(tfidf_tweet, os.path.join(SAVE_MODELS_DIR, "tfidf_tweet.joblib"))

tfidf_reply = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES_REPLY, ngram_range=(1,2))
X_tfidf_reply = tfidf_reply.fit_transform(reply_texts)  # shape (n_replies, ft_reply)
joblib.dump(tfidf_reply, os.path.join(SAVE_MODELS_DIR, "tfidf_reply.joblib"))

# Map tweet_id -> tweet row index (for tweet TF-IDF lookup)
tweetid_to_idx = {tweets_df['tweet_id'].iloc[i]: i for i in range(len(tweets_df))}

# -------------------------
# === Merge replies with parent tweet & assemble features per reply row ===
# -------------------------
print("Merging replies with parent tweets and assembling features per reply row...")
merged = pd.merge(
    replies_df[['tweet_id', 'clean_text', 'reply_compound', 'reply_len', 'reply_sentiment_label']],
    tweets_df[['tweet_id', 'clean_text', 'tweet_compound', 'tweet_len']],
    on='tweet_id', how='inner', suffixes=('_reply', '_tweet')
)

merged = merged.rename(columns={'reply_sentiment_label': 'reply_sentiment'})
merged = merged.dropna(subset=['tweet_compound', 'reply_compound']).reset_index(drop=True)
print(f"Reply-level rows: {len(merged)}, unique tweets: {merged['tweet_id'].nunique()}")

# numeric features (tweet_compound, tweet_len, optional reply_len)
numeric_cols = ['tweet_compound']
if INCLUDE_TWEET_LEN:
    numeric_cols.append('tweet_len')
if INCLUDE_REPLY_LEN:
    numeric_cols.append('reply_len')

X_numeric = merged[numeric_cols].fillna(0).values.astype(np.float32)  # (n_replies, k)

# build TF-IDF arrays aligned to merged rows:
# reply TF-IDF: simply in the same order as replies_df earlier, but merged may be subset; map by position
# Build reply index mapping: replies_df row index -> merged row index positions
# Simpler: transform merged['reply_text_clean'] using tfidf_reply.transform
X_reply_tfidf_for_merged = tfidf_reply.transform(merged['clean_text_reply'].fillna("").tolist())  # sparse (n_replies, ft_reply)

# tweet TF-IDF for each reply: lookup tweet index and get tfidf row
tweet_indices_for_merged = [tweetid_to_idx.get(tid, None) for tid in merged['tweet_id'].values]
# create matrix (n_replies, ft_tweet)
n_replies = len(merged)
ft_tweet = X_tfidf_tweet.shape[1]
X_tweet_tfidf_for_merged = csr_matrix((n_replies, ft_tweet), dtype=np.float32)
# fill row-by-row (efficient-ish)
rows = []
cols = []
data = []
for i, tid in enumerate(merged['tweet_id'].values):
    idx = tweetid_to_idx.get(tid, None)
    if idx is None:
        # leave zeros
        continue
    row_vec = X_tfidf_tweet[idx]
    # row_vec is sparse; convert to coo to iterate
    coo = row_vec.tocoo()
    rows.extend([i]*len(coo.col))
    cols.extend(coo.col.tolist())
    data.extend(coo.data.tolist())
if len(data) > 0:
    X_tweet_tfidf_for_merged = csr_matrix((data, (rows, cols)), shape=(n_replies, ft_tweet), dtype=np.float32)
else:
    X_tweet_tfidf_for_merged = csr_matrix((n_replies, ft_tweet), dtype=np.float32)

# Combine sparse TF-IDF (tweet + reply) and numeric (dense) into a single feature matrix
# We'll keep combined X as sparse for tree models; for SMOTE/NN we will convert to dense when needed.
print("Combining TF-IDF (tweet + reply) with numeric features (sparse hstack)...")
X_sparse = hstack([X_tweet_tfidf_for_merged, X_reply_tfidf_for_merged, csr_matrix(X_numeric)], format='csr')  # shape (n_replies, ft_tweet+ft_reply+k)
print("Combined sparse shape:", X_sparse.shape)

# Labels
y = merged['reply_sentiment'].values
le = LabelEncoder()
y_enc = le.fit_transform(y)
print("Class mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

# -------------------------
# === Train / Test split (stratified) ===
# -------------------------
print("Train/test split (stratified by reply sentiment)...")
# We need to keep indices for later mapping
indices = np.arange(X_sparse.shape[0])
X_train_sp, X_test_sp, y_train, y_test, idx_train, idx_test = train_test_split(
    X_sparse, y_enc, indices, test_size=0.2, random_state=42, stratify=y_enc
)

# -------------------------
# === Apply SMOTE (if chosen) or fallback to class weights ===
# -------------------------
use_smote_final = False
print("Preparing training data: attempting SMOTE if available and requested...")
if USE_SMOTE and smote_available:
    try:
        # SMOTE requires dense arrays
        print("Converting sparse training data to dense for SMOTE (may need memory)...")
        X_train_dense_for_smote = X_train_sp.toarray()
        sm = SMOTE(k_neighbors=SMOTE_K_NEIGHBORS, random_state=42)
        X_train_res, y_train_res = sm.fit_resample(X_train_dense_for_smote, y_train)
        use_smote_final = True
        print("SMOTE applied. Resampled training class distribution:", dict(pd.Series(y_train_res).value_counts()))
    except Exception as e:
        print("SMOTE failed (likely memory or other issue). Falling back to class weights. Error:", e)
        X_train_res = X_train_sp  # keep sparse
        y_train_res = y_train
else:
    X_train_res = X_train_sp
    y_train_res = y_train

# -------------------------
# === Scaling (for NN & LR) ===
# -------------------------
scaler = None
if SCALE_FEATURES:
    print("Scaling features (dense) for NN and linear models...")
    # For simplicity: convert training/resampled sets to dense (if SMOTE used it's already dense)
    if use_smote_final:
        scaler = StandardScaler()
        X_train_res = scaler.fit_transform(X_train_res)
        X_test = X_test_sp.toarray()
        X_test_scaled = scaler.transform(X_test)
    else:
        # X_train_res may be sparse; convert to dense (may be heavy). Tree models can stay sparse, but NN and LR prefer dense.
        try:
            X_train_dense = X_train_res.toarray()
            X_test_dense = X_test_sp.toarray()
            scaler = StandardScaler()
            X_train_res = scaler.fit_transform(X_train_dense)
            X_test_scaled = scaler.transform(X_test_dense)
        except Exception as e:
            # Memory error fallback: skip scaling and use sparse where possible
            print("Dense conversion for scaling failed (memory). Proceeding without scaling for NN/linear. Error:", e)
            scaler = None
            X_train_res = X_train_res  # sparse
            X_test_scaled = X_test_sp
else:
    # no scaling requested
    X_test_scaled = X_test_sp.toarray() if not hasattr(X_test_sp, "toarray") else X_test_sp.toarray()

# Compute class weights (for use if not using SMOTE)
classes_unique = np.unique(y_train_res)
cw_values = compute_class_weight(class_weight='balanced', classes=classes_unique, y=y_train_res)
class_weight_dict = {int(c): float(w) for c, w in zip(classes_unique, cw_values)}
print("Computed class weights:", class_weight_dict)

# -------------------------
# === Train classical classifiers ===
# -------------------------
print("\n--- Training classical classifiers ---")
def fit_and_eval_classical(clf, X_tr, y_tr, X_te, y_te, name):
    clf.fit(X_tr, y_tr)
    ypred = clf.predict(X_te)
    acc = accuracy_score(y_te, ypred)
    crep = classification_report(y_te, ypred, target_names=le.classes_, zero_division=0)
    cm = confusion_matrix(y_te, ypred)
    print(f"\n[{name}] Accuracy: {acc:.4f}")
    print("Classification report:")
    print(crep)
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)
    return clf, acc, crep, cm, ypred

# For classical tree models we can pass sparse matrices directly (if we didn't densify for SMOTE)
fitted_models = {}
results = {}

# Prepare training data for classical models:
if use_smote_final:
    Xtr_for_classical = X_train_res  # dense numpy
    Xte_for_classical = X_test_scaled
else:
    # prefer sparse for tree-based models; but classifiers like Logistic require dense
    Xtr_for_classical = X_train_res.toarray() if hasattr(X_train_res, "toarray") else X_train_res
    Xte_for_classical = X_test_sp.toarray() if hasattr(X_test_sp, "toarray") else X_test_sp

# Logistic (dense), RandomForest (sparse/dense ok), GradientBoosting (dense)
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=200, random_state=42)
}

for name, m in models.items():
    # For RandomForest, if we have sparse original and not SMOTE, pass sparse to benefit memory
    if name == "RandomForest" and not use_smote_final:
        # use sparse training matrix directly if available
        Xtr_rf = X_train_res if hasattr(X_train_res, "toarray") else Xtr_for_classical
        Xte_rf = X_test_sp
        try:
            clf, acc, crep, cm, ypred = fit_and_eval_classical(m, Xtr_rf, y_train if not use_smote_final else y_train_res, Xte_rf, y_test, name)
        except Exception:
            # fallback to dense
            clf, acc, crep, cm, ypred = fit_and_eval_classical(m, Xtr_for_classical, (y_train_res if use_smote_final else y_train), Xte_for_classical, y_test, name)
    else:
        Xtr = Xtr_for_classical
        Xte = Xte_for_classical
        ytr = y_train_res if use_smote_final else y_train
        clf, acc, crep, cm, ypred = fit_and_eval_classical(m, Xtr, ytr, Xte, y_test, name)
    fitted_models[name] = clf
    results[name] = {"accuracy": acc, "report": crep, "cm": cm}
    joblib.dump(clf, os.path.join(SAVE_MODELS_DIR, f"{name}.joblib"))

# -------------------------
# === Neural Network classifier (Keras) ===
# -------------------------
if tf_available:
    print("\n--- Training Neural Network (Keras) ---")
    # Determine NN training arrays (dense)
    if use_smote_final:
        Xnn_tr = X_train_res  # already dense after SMOTE
        ynn_tr = y_train_res
        Xnn_te = X_test_scaled
    else:
        # if we were able to scale & dense-convert earlier, use X_train_res (dense)
        try:
            Xnn_tr = X_train_res.toarray() if hasattr(X_train_res, "toarray") else X_train_res
            Xnn_te = X_test_scaled if isinstance(X_test_scaled, np.ndarray) else (X_test_sp.toarray() if hasattr(X_test_sp, "toarray") else X_test_sp)
            ynn_tr = y_train_res
        except Exception:
            # fallback: convert sparse to dense (may OOM)
            Xnn_tr = X_train_sp.toarray()
            Xnn_te = X_test_sp.toarray()
            ynn_tr = y_train

    input_dim = Xnn_tr.shape[1]
    num_classes = len(le.classes_)

    # Build NN
    nn = Sequential([
        Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        Dropout(0.5),
        BatchNormalization(),
        
        Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        BatchNormalization(),
        Dropout(0.4),
        
        Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        Dropout(0.3),
        
        Dense(num_classes, activation='softmax')
    ])

    def focal_loss_tf(gamma=2.0, alpha=0.25):
        import tensorflow as tf
        def loss(y_true, y_pred):
            y_true_oh = tf.one_hot(tf.cast(y_true, tf.int32), depth=tf.shape(y_pred)[-1])
            y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
            ce = -y_true_oh * tf.math.log(y_pred)
            weight = alpha * tf.pow(1 - y_pred, gamma)
            loss = weight * ce
            return tf.reduce_sum(loss, axis=1)
        return loss

    loss_fn = focal_loss_tf() if USE_FOCAL_LOSS else 'sparse_categorical_crossentropy'
    nn.compile(optimizer=Adam(learning_rate=NN_LR), loss=loss_fn, metrics=['accuracy'])

    #es = EarlyStopping(monitor='val_loss', patience=NN_PATIENCE, restore_best_weights=True, verbose=1)
    rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)

    # class weights used only if SMOTE not applied
    cw_for_nn = class_weight_dict if not use_smote_final else None

    history = nn.fit(
        Xnn_tr, ynn_tr,
        validation_split=0.1,
        epochs=500,
        batch_size=NN_BATCH,
        class_weight=cw_for_nn,
        callbacks=[rlr],
        verbose=1
    )

    loss, acc = nn.evaluate(Xnn_te, y_test, verbose=0)
    print(f"\n[NeuralNetwork] Test Accuracy: {acc:.4f}")
    ypred_nn_probs = nn.predict(Xnn_te)
    ypred_nn = np.argmax(ypred_nn_probs, axis=1)
    print("Classification report (NN):")
    print(classification_report(y_test, ypred_nn, target_names=le.classes_, zero_division=0))
    print("Confusion matrix (NN):")
    print(confusion_matrix(y_test, ypred_nn))
    nn.save(os.path.join(SAVE_MODELS_DIR, "nn_model"))
    results["NeuralNetwork"] = {"accuracy": acc, "report": classification_report(y_test, ypred_nn, target_names=le.classes_), "cm": confusion_matrix(y_test, ypred_nn)}
else:
    print("\nTensorFlow not available — skipping NN training. Install tensorflow to enable.")
    results["NeuralNetwork"] = {"accuracy": None}

# -------------------------
# === Prediction helpers ===
# -------------------------
def build_feature_for_reply(tweet_text, reply_text):
    """
    Build a single-row feature vector for a (tweet_text, reply_text) pair:
    concatenates [TF-IDF(tweet), TF-IDF(reply), numeric features].
    Returns dense 2D array shape (1, feature_dim) (scaled if scaler available).
    """
    tclean = clean_text(tweet_text)
    rclean = clean_text(reply_text)
    tf_t = tfidf_tweet.transform([tclean]).toarray()[0]
    tf_r = tfidf_reply.transform([rclean]).toarray()[0]
    num = np.array([
        get_compound(tclean),
        len(tclean) if INCLUDE_TWEET_LEN else 0,
        # reply_len optionally appended if INCLUDE_REPLY_LEN -- but we didn't include separately in this build
    ], dtype=np.float32)
    # note: earlier we placed numeric as [tweet_compound, tweet_len, (reply_len optional)]
    # ensure ordering matches training concatenation (we used X = [tweet_tfidf | reply_tfidf | X_numeric])
    feat = np.hstack([tf_t, tf_r, num])
    if scaler is not None:
        feat = scaler.transform([feat])[0]
    return feat.reshape(1, -1)

def predict_reply_sentiment_from_texts(tweet_text, reply_text, model_name="RandomForest"):
    feat = build_feature_for_reply(tweet_text, reply_text)
    if model_name == "NeuralNetwork":
        if not tf_available:
            raise RuntimeError("TensorFlow not available")
        probs = nn.predict(feat)[0]
    else:
        model = fitted_models.get(model_name)
        if model is None:
            raise ValueError(f"Model {model_name} not found. Available: {list(fitted_models.keys())}")
        probs = model.predict_proba(feat)[0]
    return dict(zip(le.inverse_transform(np.arange(len(probs))), probs))

# -------------------------
# === Save artifacts (vectorizers, scaler, label encoder) ===
# -------------------------
joblib.dump(tfidf_tweet, os.path.join(SAVE_MODELS_DIR, "tfidf_tweet.joblib"))
joblib.dump(tfidf_reply, os.path.join(SAVE_MODELS_DIR, "tfidf_reply.joblib"))
joblib.dump(le, os.path.join(SAVE_MODELS_DIR, "label_encoder.joblib"))
if scaler is not None:
    joblib.dump(scaler, os.path.join(SAVE_MODELS_DIR, "scaler.joblib"))
# save classical models already done in loop

# Save test predictions CSV for the best classical model
best_classical = max({k:v for k,v in results.items() if (k != "NeuralNetwork" and results[k]["accuracy"] is not None)}.items(), key=lambda kv: kv[1]['accuracy'])[0]
best_clf = fitted_models[best_classical]
preds_test = best_clf.predict(X_test_sp.toarray() if hasattr(X_test_sp, "toarray") else X_test_sp)
test_rows = merged.iloc[idx_test].reset_index(drop=True).copy()
test_rows['true_reply_sentiment'] = le.inverse_transform(y_test)
test_rows['pred_'+best_classical] = le.inverse_transform(preds_test)
test_rows.to_csv("test_true_pred_classifiers_with_reply_tfidf.csv", index=False, encoding="utf-8")
print(f"\nSaved test-set predictions to test_true_pred_classifiers_with_reply_tfidf.csv (best classical = {best_classical})")

print("\n--- Summary accuracies ---")
for k,v in results.items():
    print(k, "accuracy:", v['accuracy'])

print("\nDone.")


