import numpy as np
from joblib import Parallel, delayed
from xgboost import XGBClassifier


def _one_bag_iteration(X, y, pos_idx, neg_idx, n_neg, neg_sample_size, seed, model_kwargs):
    rng = np.random.default_rng(seed)
    sampled_positions = rng.choice(n_neg, size=neg_sample_size, replace=False)
    sampled_neg = neg_idx[sampled_positions]
    train_idx = np.concatenate([pos_idx, sampled_neg])

    model = XGBClassifier(random_state=seed, **model_kwargs)
    model.fit(X.iloc[train_idx], y[train_idx])

    oob_mask = np.ones(n_neg, dtype=bool)
    oob_mask[sampled_positions] = False
    oob_positions = np.where(oob_mask)[0]
    oob_preds = model.predict_proba(X.iloc[neg_idx[oob_positions]])[:, 1]
    return oob_positions, oob_preds


def run_pu_bagging(X, y, pos_idx, neg_idx, n_iter=100, neg_sample_ratio=3,
                    seed=2026, model_kwargs=None, n_jobs=-1):
    if model_kwargs is None:
        model_kwargs = dict(
            max_depth=3, learning_rate=0.05, n_estimators=100,
            min_child_weight=6, eval_metric='logloss', n_jobs=1,
        )

    n_neg = len(neg_idx)
    neg_sample_size = min(n_neg, neg_sample_ratio * len(pos_idx))

    results = Parallel(n_jobs=n_jobs)(
        delayed(_one_bag_iteration)(
            X, y, pos_idx, neg_idx, n_neg, neg_sample_size, seed + i, model_kwargs
        )
        for i in range(n_iter)
    )

    oob_score_sum = np.zeros(n_neg)
    oob_score_count = np.zeros(n_neg)
    for oob_positions, oob_preds in results:
        oob_score_sum[oob_positions] += oob_preds
        oob_score_count[oob_positions] += 1

    pu_scores = np.divide(
        oob_score_sum, oob_score_count,
        out=np.zeros_like(oob_score_sum),
        where=oob_score_count > 0
    )
    return pu_scores, oob_score_count


class PUFilterMixin:
    """
    Mix in before LOGOEvaluator, e.g.:
        class LogRegEvaluator(PUFilterMixin, LOGOEvaluator): ...
    """
    def __init__(self, *args, run_pu, pu_threshold=0.8, n_iter_fold=50, neg_sample_ratio=3, **kwargs):
        self._last_n_flagged = None
        self.run_pu = run_pu
        self.pu_threshold = pu_threshold
        self.n_iter_fold = n_iter_fold
        self.neg_sample_ratio = neg_sample_ratio
        super().__init__(*args, **kwargs)

    def transform_data_per_split(self, X, y, is_train, held_out_species):
        if not is_train or not self.run_pu:
            return X, y

        pos_idx = np.flatnonzero(y == 1)
        neg_idx = np.flatnonzero(y == 0)

        pu_scores, n_oob = run_pu_bagging(
            X, y, pos_idx, neg_idx,
            n_iter=self.n_iter_fold,
            neg_sample_ratio=self.neg_sample_ratio,
            seed=hash(held_out_species) % (2**31),
        )
        flagged_mask = pu_scores >= self.pu_threshold
        kept_neg_idx = neg_idx[~flagged_mask]
        keep_idx = np.sort(np.concatenate([pos_idx, kept_neg_idx]))

        self._last_n_flagged = int(neg_idx.size - kept_neg_idx.size)

        return X.iloc[keep_idx], y[keep_idx]
