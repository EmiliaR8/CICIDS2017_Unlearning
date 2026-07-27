# # === put this at the very top of main.py, BEFORE importing numpy/xgboost ===
# import os
# cores = os.cpu_count() or 1
# os.environ["OMP_NUM_THREADS"] = str(cores)
# os.environ["OPENBLAS_NUM_THREADS"] = str(cores)
# os.environ["MKL_NUM_THREADS"] = str(cores)
# os.environ["NUMEXPR_NUM_THREADS"] = str(cores)

# import xgboost as xgb
# import numpy as np
# import joblib
# import pandas as pd


# class XGBoostIDSWrapper:
#     def __init__(self, model=None, label_mapping=None, inverse_mapping=None, known_labels=None, feature_names=None, **kwargs):
#         """
#         :param model: Pre-trained XGBoost model (if available)
#         :param label_mapping: Dictionary mapping original labels to numeric values
#         :param inverse_mapping: Dictionary mapping numeric values back to original labels
#         :param known_labels: Set of all seen labels to prevent missing class errors
#         :param kwargs: Additional XGBoost parameters if a new model is initialized
#         """
#         # self.model = model if model is not None else xgb.XGBClassifier(**kwargs)
#         kwargs['tree_method'] = 'gpu_hist'
#         self.model = model if model is not None else xgb.XGBClassifier(**kwargs) #FIXME
        
#         self.label_mapping = label_mapping if label_mapping is not None else None
#         self.inverse_mapping = inverse_mapping if inverse_mapping is not None else None
#         self.known_labels = known_labels if known_labels is not None else set()
#         self.feature_names = feature_names


#     def fit(self, X, y, sample_weight=None):
#         """
#         Train XGBoost model while handling label encoding and missing classes.
#         """
#         #  Ensure labels are numeric
#         y_numeric = self._encode_labels(y)

#         # Track all seen labels for future updates
#         if self.known_labels is None:
#             self.known_labels = set(np.unique(y_numeric))
#         else:
#             self.known_labels.update(np.unique(y_numeric))

#         # Pass sample_weight to XGBoost if provided
#         if sample_weight is not None:
#             self.model.fit(X, y_numeric, sample_weight=sample_weight)
#         else:
#             self.model.fit(X, y_numeric)

#         return self

#     def update_m(self, X_new, y_new, num_new_trees=5):
#         """
#         Incrementally update XGBoost without full retraining.
#         1. Freeze old trees.
#         2. Add a small number of new trees trained only on the drifted data.
#         3. Ensure missing classes are represented.
#         """

#         #  Convert labels & track missing ones
#         y_new_numeric = self._encode_labels(y_new)
#         if self.known_labels:
#             all_classes = np.array(list(self.known_labels))  # Convert to NumPy array
#         else:
#             all_classes = np.unique(y_new_numeric)
#         missing_classes = set(all_classes) - set(np.unique(y_new_numeric))
#         if missing_classes:
#             print(f"️ Missing classes detected: {missing_classes}, adding dummy samples.")

#             y_expanded = np.concatenate([y_new_numeric, all_classes])
#             X_expanded = np.vstack([X_new, np.zeros((len(all_classes), X_new.shape[1]))])  # Fake features

#             # Assign ZERO weight to fake samples so they are ignored
#             sample_weights = np.concatenate([np.ones(len(X_new)), np.zeros(len(all_classes))])

#             # Train with weighted samples (ensures fake samples are ignored)
#             self.model.fit(X_expanded, y_expanded, sample_weight=sample_weights, xgb_model=self.model)
#         else:
#             num_existing_trees = self.model.get_booster().num_boosted_rounds()  # ✅ Get existing tree count
#             if num_existing_trees >= 125: #Limiting the number of trees 
#                 self.model.fit(X_new, y_new_numeric, xgb_model=self.model)  # ✅ Train on new data
#             else:
#                 self.model.n_estimators = num_existing_trees + num_new_trees  # ✅ Just increase tree count
#                 self.model.fit(X_new, y_new_numeric, xgb_model=self.model)  # ✅ Train on new data

#         '''
#         #  Update model without overwriting old trees
#         self.model = xgb.XGBClassifier(
#             n_estimators=self.model.get_booster().num_boosted_rounds() + num_new_trees,  # ✅ Add new trees
#             learning_rate=self.model.learning_rate,
#             max_depth=self.model.max_depth,
#             objective=self.model.objective,
#             booster=self.model.booster,
#             tree_method=self.model.tree_method
#         )

#         # ✅ Train only on new + dummy samples, keeping old model
#         self.model.fit(X_new, y_new_numeric, xgb_model=self.model)
#         '''
#         return self

#     def predict(self, X):
#         preds_numeric = self.model.predict(X)
#         return self._decode_labels(preds_numeric)

#     def predict_proba(self, X):
#         return self.model.predict_proba(X)

#     def get_booster(self):
#         return self.model.get_booster()

#     def save(self, path):
#         joblib.dump({
#             'model': self.model,
#             'label_mapping': self.label_mapping,
#             'inverse_mapping': self.inverse_mapping,
#             'known_labels': self.known_labels  # ✅ Save known labels
#         }, path)

#     @classmethod
#     def load(cls, path):
#         data = joblib.load(path)
#         instance = cls()
#         instance.model = data['model']
#         instance.label_mapping = data['label_mapping']
#         instance.inverse_mapping = data['inverse_mapping']
#         instance.known_labels = data['known_labels']  # ✅ Load known labels
#         return instance

#     def _encode_labels(self, y):
#         """
#         Encode string labels into numeric labels for XGBoost.
#         """
#         if isinstance(y[0], str):
#             if self.label_mapping is None:
#                 unique_labels = sorted(np.unique(y))
#                 self.label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
#                 self.inverse_mapping = {v: k for k, v in self.label_mapping.items()}
#             return np.array([self.label_mapping[label] for label in y])
#         return y

#     def _decode_labels(self, preds_numeric):
#         """
#         Decode numeric predictions back to original string labels.
#         """
#         if self.inverse_mapping is not None:
#             return np.array([self.inverse_mapping[pred] for pred in preds_numeric])
#         return preds_numeric


# ===================== XGBoostIDSWrapper (Model A: single process, all cores) =====================
# If possible, set these env vars at the very top of your main entrypoint BEFORE importing numpy/xgboost.
import os
cores = os.cpu_count() or 1
os.environ.setdefault("OMP_NUM_THREADS", str(cores))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(cores))
os.environ.setdefault("MKL_NUM_THREADS", str(cores))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(cores))

import xgboost as xgb
import numpy as np
import joblib
import time

class XGBoostIDSWrapper:
    def __init__(self, model=None, label_mapping=None, inverse_mapping=None, known_labels=None, feature_names=None, **kwargs):
        """
        :param model: Pre-trained XGBoost model (sklearn wrapper) if available
        :param kwargs: Additional XGBoost parameters if a new model is initialized
        """
        # Enforce CPU parallelism for Model A
        kwargs.setdefault("tree_method", "hist")
        kwargs.setdefault("predictor", "cpu_predictor")
        kwargs.setdefault("max_depth", 6)
        kwargs.setdefault("max_bin", 256)
        kwargs.setdefault("subsample", 0.8)
        kwargs.setdefault("colsample_bytree", 0.8)
        # Let training set the exact thread count; safe default here:
        kwargs.setdefault("nthread", cores)

        self.model = model if model is not None else xgb.XGBClassifier(**kwargs)

        self.label_mapping = label_mapping if label_mapping is not None else None
        self.inverse_mapping = inverse_mapping if inverse_mapping is not None else None
        self.known_labels = known_labels if known_labels is not None else set()
        self.feature_names = feature_names

    # ------------------- Public API -------------------

    def fit(self, X, y, sample_weight=None):
        y_numeric = self._encode_labels(y)
        if self.known_labels is None:
            self.known_labels = set(np.unique(y_numeric))
        else:
            self.known_labels.update(np.unique(y_numeric))
        if sample_weight is not None:
            self.model.fit(X, y_numeric, sample_weight=sample_weight)
        else:
            self.model.fit(X, y_numeric)
        return self

    def update_m(self, X_new, y_new, num_new_trees=16, max_trees_total=800):
        """Backwards-compatible name; forwards to incremental_step."""
        return self.incremental_step(X_new, y_new, num_new_trees=num_new_trees, max_trees_total=max_trees_total)

    def incremental_step(self, X_new, y_new, num_new_trees=16, max_trees_total=800):
        """
        Fast incremental update:
          1) Refresh existing trees' leaves (no new trees).
          2) Grow up to `num_new_trees`, but never exceed `max_trees_total`.
        This avoids slow creep and accidental full retrains.
        """
        import gc, time

        # Ensure fast, contiguous floats
        X_new = np.asarray(X_new, dtype=np.float32, order="C")
        y_new = self._encode_labels(y_new)

        # Parameters for this update (Model A: use all cores)
        params = self.model.get_xgb_params() if hasattr(self.model, "get_xgb_params") else {}
        params.update({
            "tree_method": "hist",
            "predictor": "cpu_predictor",
            "nthread": os.cpu_count() or 1,
            "max_depth": params.get("max_depth", 6),
            "max_bin": params.get("max_bin", 256),
            "subsample": params.get("subsample", 0.8),
            "colsample_bytree": params.get("colsample_bytree", 0.8),
        })

        # Build DMatrix (faster than sklearn fit for small updates)
        t0 = time.time()
        dtrain = xgb.DMatrix(X_new, label=y_new, feature_names=self.feature_names)
        # print(f"[xgb] dmatrix in {time.time()-t0:.3f}s", flush=True)

        # Get current booster
        booster = self.model.get_booster() if hasattr(self.model, "get_booster") else self.model._Booster

        # Keep sklearn wrapper synchronized to prevent full retrains
        if hasattr(self.model, "_Booster"):
            self.model._Booster = booster
        if hasattr(self.model, "n_estimators"):
            try:
                self.model.n_estimators = booster.num_boosted_rounds()
            except Exception:
                pass

        # Step 1: Refresh (fast)
        rounds_before = booster.num_boosted_rounds()
        if rounds_before > 0:
            refresh_params = dict(params, process_type="update", updater="refresh", refresh_leaf=True)
            t1 = time.time()
            xgb.train(refresh_params, dtrain, num_boost_round=0, xgb_model=booster)
            # print(f"[xgb] refresh in {time.time()-t1:.3f}s (trees={rounds_before})", flush=True)

        # Step 2: Bounded grow
        before = booster.num_boosted_rounds()
        to_add = max(0, min(num_new_trees, max_trees_total - before))
        if to_add > 0:
            t2 = time.time()
            updated = xgb.train(params, dtrain, num_boost_round=to_add, xgb_model=booster)

            # stitch updated booster back into sklearn wrapper
            if hasattr(self.model, "_Booster"):
                self.model._Booster = updated
            if hasattr(self.model, "n_estimators"):
                self.model.n_estimators = updated.num_boosted_rounds()
            booster = updated
            # print(f"[xgb] grow {to_add} in {time.time()-t2:.3f}s (now={booster.num_boosted_rounds()})", flush=True)

        # Optional: quick predict timing to catch stalls
        t3 = time.time()
        _ = booster.predict(dtrain)
        # print(f"[xgb] predict in {time.time()-t3:.3f}s", flush=True)

        del dtrain
        gc.collect()
        return self

    def predict(self, X):
        preds_numeric = self.model.predict(X)
        return self._decode_labels(preds_numeric)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def get_booster(self):
        return self.model.get_booster()

    def save(self, path):
        joblib.dump({
            'model': self.model,
            'label_mapping': self.label_mapping,
            'inverse_mapping': self.inverse_mapping,
            'known_labels': self.known_labels
        }, path)

    @classmethod
    def load(cls, path):
        data = joblib.load(path)
        instance = cls()
        instance.model = data['model']
        instance.label_mapping = data['label_mapping']
        instance.inverse_mapping = data['inverse_mapping']
        instance.known_labels = data['known_labels']
        return instance

    # ------------------- Encoding helpers -------------------

    def _encode_labels(self, y):
        if isinstance(y[0], str):
            if self.label_mapping is None:
                unique_labels = sorted(np.unique(y))
                self.label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
                self.inverse_mapping = {v: k for k, v in self.label_mapping.items()}
            return np.array([self.label_mapping[label] for label in y])
        return y

    def _decode_labels(self, preds_numeric):
        if self.inverse_mapping is not None:
            return np.array([self.inverse_mapping[pred] for pred in preds_numeric])
        return preds_numeric
# ===================== end wrapper =====================
