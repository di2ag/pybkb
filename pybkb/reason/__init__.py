from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as sk_precision_recall_fscore_support

class BaseReasoner:
    def __init__(self):
        pass

    @staticmethod
    def _get_labels(true, pred):
        return list(set.union(*[set(true), set(pred)]))

    @staticmethod
    def confusion_matrix(true, pred, labels=None):
        if labels is None:
            labels = BaseReasoner._get_labels(true, pred)
        return sk_confusion_matrix(true, pred, labels=labels), labels

    @staticmethod
    def precision_recall_fscore_support(true, pred, labels=None, average=None):
        # Check to make sure there aren't any no predictions from bkb, don't want to include these in metrics
        new_true, new_pred, no_inference = [], [], []
        for idx, (true_value, pred_value) in enumerate(zip(true, pred)):
            if pred_value is None:
                no_inference.append(idx)
                continue
            new_true.append(true_value)
            new_pred.append(pred_value)
        if labels is None:
            labels = BaseReasoner._get_labels(new_true, new_pred)
        return sk_precision_recall_fscore_support(new_true, new_pred, labels=labels, average=average, zero_division=0), labels, no_inference
