import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import matplotlib.pyplot as plt
import joblib

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# ---------- Config ----------
CSV_PATH = "greenhouse_data_.csv"
FEATURE_COLS = ['temp_c', 'rh', 'light_lux', 'soil_moisture', 'co2']

# ---------- Utility: synthetic generator ----------
def generate_synthetic_data(n=10000):
    temps = np.random.normal(loc=24, scale=4, size=n)
    rh = np.clip(np.random.normal(loc=60, scale=12, size=n), 10, 100)
    light = np.clip(np.random.normal(loc=20000, scale=12000, size=n), 0, None)
    soil = np.clip(np.random.normal(loc=45, scale=20, size=n), 0, 100)
    co2 = np.clip(np.random.normal(loc=450, scale=80, size=n), 200, None)

    actions = []
    for t, h, l, s, c in zip(temps, rh, light, soil, co2):
        if t > 28 and h < 60:
            actions.append("FAN")
        elif s < 30:
            actions.append("IRRIGATE")
        elif l > 50000:
            actions.append("SHADE")
        else:
            actions.append("NONE")

    df = pd.DataFrame({
        'temp_c': temps.round(2),
        'rh': rh.round(2),
        'light_lux': light.round(2),
        'soil_moisture': soil.round(2),
        'co2': co2.round(2),
        'target_action': actions
    })
    return df

# ---------- 0) Load dataset ----------
if os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded CSV: {CSV_PATH}")
else:
    print(f"CSV not found. Using synthetic data instead.")
    df = generate_synthetic_data(n=12000)

print("Dataset shape:", df.shape)
print("Columns:", list(df.columns))
print(df.head(5))

# ---------- 1) Basic validation ----------
missing_features = [c for c in FEATURE_COLS if c not in df.columns]
if missing_features:
    raise ValueError(f"Missing feature columns: {missing_features}")

# ---------- 2) Classification pipeline ----------
def prepare_classification(df):
    X = df[FEATURE_COLS].values
    if 'target_action' not in df.columns:
        raise ValueError("Missing 'target_action' column")
    le = LabelEncoder()
    y = le.fit_transform(df['target_action'].values)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=RANDOM_SEED, stratify=y)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, scaler, le

X_train, X_test, y_train, y_test, clf_scaler, label_enc = prepare_classification(df)

# ---------- 3) Classification model ----------
def build_classification_model(input_dim, n_classes):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dense(n_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

clf_model = build_classification_model(X_train.shape[1], n_classes=len(label_enc.classes_))
clf_model.summary()

es = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history_clf = clf_model.fit(X_train, y_train, validation_split=0.15,
                            epochs=100, batch_size=64, callbacks=[es], verbose=2)

# ---------- 4) Evaluate classification ----------
print("\nClassification evaluation on test set:")
clf_loss, clf_acc = clf_model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {clf_loss:.4f}, Test Acc: {clf_acc:.4f}")

y_pred_probs = clf_model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_enc.classes_))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ---------- 5) Plot training history ----------
def plot_history(history, title_suffix=""):
    if history is None:
        return
    plt.figure(figsize=(8,4))
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title(f'Loss {title_suffix}')
    plt.xlabel('epoch')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_history(history_clf, "(classification)")

# ---------- 6) Save model & scaler ----------
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
clf_model.save(os.path.join(MODEL_DIR, "greenhouse_classifier"))
joblib.dump(clf_scaler, os.path.join(MODEL_DIR, "clf_scaler.joblib"))
joblib.dump(label_enc, os.path.join(MODEL_DIR, "label_encoder.joblib"))
print(f"\nModel and scaler saved to {MODEL_DIR}/")

# ---------- 7) Example inference ----------
def infer_classification(observation_dict):
    obs = np.array([[observation_dict[c] for c in FEATURE_COLS]])
    obs_scaled = clf_scaler.transform(obs)
    model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "greenhouse_classifier"))
    probs = model.predict(obs_scaled)[0]
    pred_idx = int(np.argmax(probs))
    label = label_enc.inverse_transform([pred_idx])[0]
    return label, probs

# Quick test inference
sample_obs = {
    'temp_c': 30.0,
    'rh': 45.0,
    'light_lux': 30000,
    'soil_moisture': 25.0,
    'co2': 420.0
}
print("\nSample inference (classification):", infer_classification(sample_obs))

if __name__ == "__main__":
    print("\nAll done.")
