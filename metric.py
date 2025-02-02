from typing import Literal, Optional, Union

import numpy as np

import torch
import torch.nn.functional as F
from sklearn.metrics.cluster import contingency_matrix
import torchmetrics.functional as metrics



def predictive_scores(y_pred: torch.Tensor, y_true: torch.Tensor, 
                      task: Literal["binary", "multiclass", "multilabel"], 
                      num_classes: Optional[int] = None,
                      pred_threshold: float = 0.5, 
                      average="none"):
    if y_true.ndim > 1 and task == "multiclass":
        y_true = torch.argmax(y_true, dim=-1)

    if num_classes is None:
        num_classes = np.max(y_true) if task == "multiclass" else y_true.size(-1)

    auc_family_args = dict(
        preds=y_pred, 
        target=y_true,
        task=task, 
        num_classes=num_classes, 
        num_labels=num_classes,
        average=average
    )
    AUROC = metrics.auroc(**auc_family_args).numpy(force=True)
    AUPRC = metrics.average_precision(**auc_family_args).numpy(force=True)

    f1_family_args = dict(
        preds=y_pred, 
        target=y_true,
        task=task, 
        num_classes=num_classes,
        num_labels=num_classes,
        threshold=pred_threshold,
        average=average
    )
    F1   = metrics.f1_score(**f1_family_args).numpy(force=True)
    PREC = metrics.precision(**f1_family_args).numpy(force=True)
    REC  = metrics.recall(**f1_family_args).numpy(force=True)

    # calculate class-wise MCC
    if average == "none":
        if task == "multiclass":
            y_pred_mcc = F.one_hot(torch.argmax(y_pred, dim=-1), num_classes=num_classes)
        else:
            y_pred_mcc = (y_pred >= pred_threshold).to(dtype=torch.int64)
        MCC = np.zeros(num_classes)
        for c in range(num_classes):
            mcc_args = dict(
                preds=y_pred_mcc[:, c], 
                target=(y_true == c).int() if task == "multiclass" else y_true[:, c],
                task="binary",
                num_classes=num_classes,
                num_labels=num_classes,
                threshold=pred_threshold
            )
            MCC[c] = metrics.matthews_corrcoef(**mcc_args).item()
    else:
        mcc_args = dict(
            preds=y_pred, 
            target=y_true,
            task=task,
            num_classes=num_classes,
            num_labels=num_classes,
            threshold=pred_threshold
        )
        MCC = metrics.matthews_corrcoef(**mcc_args).numpy(force=True)
    
    return MCC, F1, PREC, REC, AUROC, AUPRC

def hamming_score(y_true: np.ndarray, y_pred: np.ndarray, average: Optional[str]=None):
    numerator = (y_true & y_pred).sum(axis=1)
    denominator = (y_true | y_pred).sum(axis=1)
    scores = np.divide(numerator, denominator, out=np.ones_like(numerator, dtype=np.float_), where=denominator != 0)

    if average == 'mean':
        return scores.mean()
    else:
        return scores

def purity_score(y_pred: Union[np.ndarray, torch.Tensor], 
                 y_true: Union[np.ndarray, torch.Tensor],
                 num_classes: int, 
                 single_label=True) -> Union[np.ndarray, torch.Tensor]:
    if isinstance(y_pred, torch.Tensor) and isinstance(y_true, torch.Tensor):
        C = num_classes
        eval_task = 'multiclass' if single_label else 'multilabel'
        c_matrix = metrics.confusion_matrix(y_pred, y_true, num_classes=C, num_labels=C, task=eval_task)
        return torch.sum(torch.amax(c_matrix, dim=0)) / torch.sum(c_matrix)
    else:
        c_matrix = contingency_matrix(y_true, y_pred)
        return np.sum(np.amax(c_matrix, axis=0)) / np.sum(c_matrix)

    # c_matrix = contingency_matrix(y_true.numpy(force=True), y_pred.numpy(force=True))

    # return np.sum(np.amax(c_matrix, axis=0)) / np.sum(c_matrix)
    # 0.3244
