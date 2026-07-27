import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, classification_report, accuracy_score
from model import XGBoostIDSWrapper

def update_xgboost(model, X_new, y_new, num_new_trees=10):
    """
    Update XGBoost without full retraining.
    1. Freeze old trees.
    2. Add a small number of new trees trained only on the drifted data.
    """

    #  Keep all old trees, only add new ones
    new_model = xgb.XGBClassifier(
        n_estimators=model.get_booster().num_boosted_rounds() + num_new_trees,  # Add new trees
        learning_rate=model.learning_rate,
        max_depth=model.max_depth,
        objective=model.objective,
        booster=model.booster,
        tree_method='gpu_hist',       #FIXME
        predictor='gpu_predictor',           #FIXME
    )

    #  Train ONLY on the new data
    new_model.fit(X_new, y_new)
    return new_model  #  Updated model

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    malicious_ratio = np.clip(np.mean(y), 1e-5, 1 - 1e-5)

    label_counts = Counter(y_train)
    print("\n======= Classifier training set label distribution =======")
    for label, count in sorted(label_counts.items()):
        print(f"Class {label}: {count}")

    label_counts = Counter(y_test)
    print("\n======= Classifier testing set label distribution =======")
    for label, count in sorted(label_counts.items()):
        print(f"Class {label}: {count}")

    # print("\nClassifier training set label distribution:\n", pd.Series(y_train).value_counts())
    # print("Classifer test set label distribution:\n", pd.Series(y_test).value_counts())

    # Initialize the wrapper with desired parameters
    model = XGBoostIDSWrapper(
        objective='binary:logistic', 
        n_estimators=10,
        use_label_encoder=False, 
        eval_metric='logloss', 
        base_score = malicious_ratio,
    )

    # Fit the model; the wrapper handles label conversion if needed
    model.fit(X_train, y_train)
    

    y_pred = model.predict(X_test)
    acc = balanced_accuracy_score(y_test, y_pred)
    accy = accuracy_score(y_test, y_pred)
    print("\nClassifier testing accuracy: ", acc)
    print("<3 Initial Un-balanced classifier testing accuracy: ", accy)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    # breakpoint()
    
    return model, acc
