import numpy as np

class EnsembleEvaluatorWrapper:
    """
    Wraps trained model to provide both prediction and epistemic uncertainty.
    For regression: returns (pred_mean, pred_std)
    For classification: returns (pred_prob_class1, pred_std)
    """
    def __init__(self, model, prop_type, prop_name):
        self.model = model
        self.prop_type = prop_type
        self.prop_name = prop_name

    def predict_with_uncertainty(self, X):
        if self.prop_type == "regression":
            all_preds = np.array([tree.predict(X) for tree in self.model.estimators_])
            pred_mean = np.mean(all_preds, axis=0)
            pred_std = np.std(all_preds, axis=0)
            return pred_mean, pred_std
        else:
            prob_class_1 = self.model.predict_proba(X)[:, 1]
            all_probs = np.array([tree.predict_proba(X)[:, 1] for tree in self.model.estimators_])
            prob_std = np.std(all_probs, axis=0)
            return prob_class_1, prob_std
