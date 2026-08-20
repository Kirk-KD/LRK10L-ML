from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, average_precision_score, log_loss, f1_score


def precision_recall(y_true, y_pred_binary):
    tp = int(((y_true == 1) & y_pred_binary).sum())
    fp = int(((y_true == 0) & y_pred_binary).sum())
    fn = int(((y_true == 1) & ~y_pred_binary).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
    recall = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
    return precision, recall


def best_f1_threshold(y_true, scores):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    candidates = np.unique(scores)
    best_t, best_f1 = candidates[0], -1.0
    for t in candidates:
        f1 = f1_score(y_true, (scores >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    return float(best_t), float(best_f1)


def get_validation_species(all_species, test_species):
    """Deterministically pick one non-test species as the validation species,
    cyclically, so validation duty rotates across folds reproducibly."""
    others = sorted(s for s in all_species if s != test_species)
    idx = sorted(all_species).index(test_species) % len(others)
    return others[idx]


class LOGOEvaluator:
    def __init__(self, logo, X, y, groups, auto_threshold=True,
                 X_full=None, y_full=None, groups_full=None):
        self.logo = logo
        self.X = X
        self.y = y
        self.groups = groups
        self.auto_threshold = auto_threshold
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

        all_species = sorted(set(self.groups))

        for train_idx, test_idx in pbar:
            test_species = self.groups[test_idx][0]
            test_species_pretty = test_species.split('_')[0]
            pbar.set_postfix_str(test_species_pretty)

            validation_species = get_validation_species(all_species, test_species)

            groups_train = self.groups[train_idx]
            fit_mask = groups_train != validation_species
            valspecies_mask = groups_train == validation_species

            fit_idx = train_idx[fit_mask]
            valspecies_idx = train_idx[valspecies_mask]

            X_fit, y_fit = self.transform_data_per_split(
                self.X.iloc[fit_idx], self.y[fit_idx],
                is_train=True, held_out_species=test_species,
            )
            X_valspecies, y_valspecies = self.transform_data_per_split(
                self.X.iloc[valspecies_idx], self.y[valspecies_idx],
                is_train=False, held_out_species=validation_species,
            )
            X_test, y_test = self.transform_data_per_split(
                self.X.iloc[test_idx], self.y[test_idx],
                is_train=False, held_out_species=test_species,
            )

            model = self.make_model_per_split(X_fit, y_fit)
            model = self.fit_model_per_split(model, X_fit, y_fit, X_valspecies, y_valspecies)

            fit_preds = model.predict_proba(X_fit)[:, 1]
            valspecies_preds = model.predict_proba(X_valspecies)[:, 1]
            test_preds = model.predict_proba(X_test)[:, 1]

            if self.auto_threshold:
                threshold, val_f1 = best_f1_threshold(y_valspecies, valspecies_preds)
            else:
                threshold, val_f1 = 0.5, float('nan')

            fit_pr_auc = average_precision_score(y_fit, fit_preds)
            test_pr_auc = average_precision_score(y_test, test_preds)

            fit_pred_binary = fit_preds >= threshold
            test_pred_binary = test_preds >= threshold

            fit_precision, fit_recall = precision_recall(y_fit, fit_pred_binary)
            test_precision, test_recall = precision_recall(y_test, test_pred_binary)

            fit_accuracy = accuracy_score(y_fit, fit_pred_binary)
            test_accuracy = accuracy_score(y_test, test_pred_binary)

            fit_bias = fit_preds.mean() - y_fit.mean()
            test_bias = test_preds.mean() - y_test.mean()

            fit_logloss = log_loss(y_fit, fit_preds, labels=[0, 1])
            test_logloss = log_loss(y_test, test_preds, labels=[0, 1])

            baseline = float(y_test.mean())
            lift = test_pr_auc / baseline if baseline > 0 else float('nan')

            X_test_full = y_test_full = preds_full = None
            if self.X_full is not None:
                full_mask = self.groups_full == test_species
                X_test_full = self.X_full[full_mask]
                y_test_full = self.y_full[full_mask]

                preds_full = model.predict_proba(X_test_full)[:, 1]
                predicted_positive_full = preds_full >= threshold

                found = int(((y_test_full == 1) & predicted_positive_full).sum())
                false_positives = int(((y_test_full == 0) & predicted_positive_full).sum())
                n_pos = int(y_test_full.sum())

                self.fold_details[test_species] = pd.DataFrame({
                    'id': X_test_full.index,
                    'y_true': y_test_full,
                    'pred': preds_full,
                }).sort_values('pred', ascending=False).reset_index(drop=True)
            else:
                found = int(((y_test == 1) & test_pred_binary).sum())
                false_positives = int(((y_test == 0) & test_pred_binary).sum())
                n_pos = int(y_test.sum())

            extra = self.extra_metrics_per_split(model, X_fit, y_fit, X_test, y_test)

            self.fold_results[test_species] = {
                'species': test_species_pretty,
                'validation_species': validation_species.split('_')[0],
                'threshold': threshold,
                'val_f1_at_threshold': val_f1,
                'baseline': baseline,
                'lift': lift,
                'fit_pr_auc': fit_pr_auc,
                'test_pr_auc': test_pr_auc,
                'gap': fit_pr_auc - test_pr_auc,
                'log_loss_gap': test_logloss - fit_logloss,
                'fit_precision': fit_precision,
                'test_precision': test_precision,
                'fit_recall': fit_recall,
                'test_recall': test_recall,
                'fit_accuracy': fit_accuracy,
                'test_accuracy': test_accuracy,
                'fit_bias': fit_bias,
                'test_bias': test_bias,
                'n_pos': n_pos,
                'found': found,
                'false_positives': false_positives,
                **extra,
            }
            self.fold_preds[test_species] = {'y_true': y_test, 'preds': test_preds}
            self.fold_models[test_species] = model
            self.fold_artifacts[test_species] = self.extra_artifacts_per_split(
                model, X_fit, y_fit, X_test, y_test, X_test_full, y_test_full, preds_full
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
