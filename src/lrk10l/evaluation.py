from typing import Any

import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, average_precision_score, log_loss


def precision_recall(y_true, y_pred_binary):
    tp = int(((y_true == 1) & y_pred_binary).sum())
    fp = int(((y_true == 0) & y_pred_binary).sum())
    fn = int(((y_true == 1) & ~y_pred_binary).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
    recall = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
    return precision, recall


class LOGOEvaluator:
    def __init__(self, logo, X, y, groups, threshold=0.5, X_full=None, y_full=None, groups_full=None):
        self.logo = logo
        self.X = X
        self.y = y
        self.groups = groups
        self.threshold = threshold
        self.X_full = X_full
        self.y_full = y_full
        self.groups_full = groups_full

        self.fold_results = {}
        self.fold_preds = {}
        self.fold_models = {}
        self.fold_artifacts = {}
        self.fold_details = {}

    def fit(self):
        n_splits = self.logo.get_n_splits(self.X, self.y, groups=self.groups)
        pbar = tqdm(self.logo.split(self.X, self.y, groups=self.groups), total=n_splits, desc="Evaluating")

        for train_idx, val_idx in pbar:
            held_out_species = self.groups[val_idx][0]
            held_out_species_pretty = held_out_species.split('_')[0]
            pbar.set_postfix_str(held_out_species_pretty)

            X_train, y_train = self.transform_data_per_split(
                self.X.iloc[train_idx], self.y[train_idx],
                is_train=True, held_out_species=held_out_species,
            )
            X_val, y_val = self.transform_data_per_split(
                self.X.iloc[val_idx], self.y[val_idx],
                is_train=False, held_out_species=held_out_species,
            )

            model = self.make_model_per_split(X_train, y_train)
            model = self.fit_model_per_split(model, X_train, y_train, X_val, y_val)

            train_preds = model.predict_proba(X_train)[:, 1]
            val_preds = model.predict_proba(X_val)[:, 1]

            train_pr_auc = average_precision_score(y_train, train_preds)
            val_pr_auc = average_precision_score(y_val, val_preds)

            train_pred_binary = train_preds >= self.threshold
            val_pred_binary = val_preds >= self.threshold

            train_precision, train_recall = precision_recall(y_train, train_pred_binary)
            val_precision, val_recall = precision_recall(y_val, val_pred_binary)

            train_accuracy = accuracy_score(y_train, train_pred_binary)
            val_accuracy = accuracy_score(y_val, val_pred_binary)

            train_bias = train_preds.mean() - y_train.mean()
            val_bias = val_preds.mean() - y_val.mean()

            train_logloss = log_loss(y_train, train_preds, labels=[0, 1])
            val_logloss = log_loss(y_val, val_preds, labels=[0, 1])

            baseline = float(y_val.mean())
            lift = val_pr_auc / baseline if baseline > 0 else float('nan')

            X_val_full = y_val_full = preds_full = None
            if self.X_full is not None:
                full_mask = self.groups_full == held_out_species
                X_val_full = self.X_full[full_mask]
                y_val_full = self.y_full[full_mask]

                preds_full = model.predict_proba(X_val_full)[:, 1]
                predicted_positive_full = preds_full >= self.threshold

                found = int(((y_val_full == 1) & predicted_positive_full).sum())
                false_positives = int(((y_val_full == 0) & predicted_positive_full).sum())
                n_pos = int(y_val_full.sum())

                self.fold_details[held_out_species] = pd.DataFrame({
                    'id': X_val_full.index,
                    'y_true': y_val_full,
                    'pred': preds_full,
                }).sort_values('pred', ascending=False).reset_index(drop=True)
            else:
                found = int(((y_val == 1) & val_pred_binary).sum())
                false_positives = int(((y_val == 0) & val_pred_binary).sum())
                n_pos = int(y_val.sum())

            extra = self.extra_metrics_per_split(model, X_train, y_train, X_val, y_val)

            self.fold_results[held_out_species] = {
                'species': held_out_species_pretty,
                'baseline': baseline,
                'lift': lift,
                'train_pr_auc': train_pr_auc,
                'val_pr_auc': val_pr_auc,
                'gap': train_pr_auc - val_pr_auc,
                'log_loss_gap': val_logloss - train_logloss,
                'train_precision': train_precision,
                'val_precision': val_precision,
                'train_recall': train_recall,
                'val_recall': val_recall,
                'train_accuracy': train_accuracy,
                'val_accuracy': val_accuracy,
                'train_bias': train_bias,
                'val_bias': val_bias,
                'n_pos': n_pos,
                'found': found,
                'false_positives': false_positives,
                **extra,
            }
            self.fold_preds[held_out_species] = {'y_true': y_val, 'preds': val_preds}
            self.fold_models[held_out_species] = model
            self.fold_artifacts[held_out_species] = self.extra_artifacts_per_split(
                model, X_train, y_train, X_val, y_val, X_val_full, y_val_full, preds_full
            )

        return self

    @property
    def results_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.fold_results).T

    def make_model_per_split(self, X_train, y_train) -> Any:
        raise NotImplementedError

    def fit_model_per_split(self, model, X_train, y_train, X_val, y_val) -> Any:
        model.fit(X_train, y_train)
        return model

    def transform_data_per_split(self, X, y, is_train, held_out_species):
        return X, y

    def extra_metrics_per_split(self, model, X_train, y_train, X_val, y_val) -> dict:
        return {}

    def extra_artifacts_per_split(self, model, X_train, y_train, X_val, y_val,
                                   X_val_full, y_val_full, preds_full) -> dict:
        return {}
