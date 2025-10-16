import multiprocessing
import os
import random
import threading

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

matplotlib.use('agg')  
print("Number of CPUs detected by Python:", multiprocessing.cpu_count())
print("OMP_NUM_THREADS:", os.environ.get("OMP_NUM_THREADS"))
print("MKL_NUM_THREADS:", os.environ.get("MKL_NUM_THREADS"))

seed = random.randint(1, 50)
print(f"Seed: {seed}")

for file in ['artist_train_tanh', 'artist_train_fixed']:
    df = pd.read_csv(file+'.csv')
    feature_columns = df.columns[2:]
    label_column = df.columns[1]
    X = df[feature_columns].values
    y = df[label_column].values

    # split by row index
    all_idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        all_idx, test_size=0.2, random_state=42, stratify=df[label_column]
    )

    y_train, y_test = df[label_column].iloc[train_idx], df[label_column].iloc[test_idx]
    X_train, X_test = df[feature_columns].iloc[train_idx].values, df[feature_columns].iloc[test_idx].values

    rf = RandomForestClassifier(n_estimators=10000, random_state=seed, n_jobs=-1)

    def train():
        rf.fit(X_train, y_train)

    t = threading.Thread(target=train)
    t.start()

    t.join()
    y_pred = rf.predict(X_test)

    joblib.dump(rf, "./RandomForest/"+file+".joblib")
    print("Model saved to ./RandomForest/"+file+".joblib")

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("\nFeature ranking:")
    for f in range(len(feature_columns)):
        print(f"{f + 1}. {feature_columns[indices[f]]} ({importances[indices[f]]:.4f})")

    plt.figure(figsize=(12,6))
    plt.title("Feature Importances")
    plt.bar(range(len(feature_columns)), importances[indices], align='center')
    plt.xticks(range(len(feature_columns)), [feature_columns[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.savefig(f"feature_importances "+file+" {seed}.png")