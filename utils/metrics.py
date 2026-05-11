import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score
from transformers import EvalPrediction

def get_precision_at_n(gt_labels, pred, k=5):
    """
    Compute precision at k
    :param gt_labels: array of ground truth labels
    :param pred: array of predicted labels
    :param k: number of top predictions to consider
    :return: score of precision at k
    """
    scores = []
    # print(gt_labels)
    # print(pred)
    for gt_row, row in zip(gt_labels, pred):
        # print(gt_row)
        # print(row)
        order = np.argsort(row)[::-1]
        # print(order)
        pred = set(order[:k])
        # print(pred)
        gt = set(np.where(gt_row==1)[0])
        # print(gt)

        if len(gt)>0:
            numenator = len(gt.intersection(pred))
            score = numenator/min(k, len(gt))
            scores.append(score)
    return np.mean(scores)


def multi_label_metrics(predictions, labels, threshold=0.3,apply_sigmoid=True):
    """
    Compute multi-label metrics for multi-label classification
    :param predictions: predicted labels
    :param labels: ground truth labels
    :param threshold: threshold for considering predictions equal to 1
    :return: a dictionary of metrics
    """
    # print(predictions)
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)

    if apply_sigmoid:
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(torch.Tensor(predictions))
    else:
        probs=torch.Tensor(predictions)
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    # print(y_pred[0])
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels
    precision_at_1=get_precision_at_n(labels,probs.numpy(),k=1)
    precision_at_5=get_precision_at_n(labels, probs.numpy() )
    precision_at_10=get_precision_at_n(labels, probs.numpy() ,k=10)
    precision_at_20=get_precision_at_n(labels, probs.numpy() ,k=20)


    metrics=multi_label_metrics_binary(y_true, y_pred)
    metrics['precision_at_1']=precision_at_1
    metrics['precision_at_5']=precision_at_5
    metrics['precision_at_10'] = precision_at_10
    metrics['precision_at_20'] = precision_at_20

    # print(metrics)
    return metrics

def multi_label_metrics_binary(y_true,y_pred):
    """
    Compute multi-label metrics for multi-label classification
    :param y_true: ground truth labels
    :param y_pred: predicted labels
    :return: a dictionary of metrics
    """
    precision_micro_average = precision_score(y_true=y_true, y_pred=y_pred, average='micro')
    precision_macro_average = precision_score(y_true=y_true, y_pred=y_pred, average='macro')
    precision_weighted_average = precision_score(y_true=y_true, y_pred=y_pred, average='weighted')
    precision_samples_average = precision_score(y_true=y_true, y_pred=y_pred, average='samples')
    precision_per_class = precision_score(y_true=y_true, y_pred=y_pred, average=None)





    recall_micro_average = recall_score(y_true=y_true, y_pred=y_pred, average='micro')
    recall_macro_average = recall_score(y_true=y_true, y_pred=y_pred, average='macro')
    recall_weighted_average = recall_score(y_true=y_true, y_pred=y_pred, average='weighted')
    recall_samples_average = recall_score(y_true=y_true, y_pred=y_pred, average='samples')
    recall_per_class = recall_score(y_true=y_true, y_pred=y_pred, average=None)

    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    f1_macro_average = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
    f1_weighted_average = f1_score(y_true=y_true, y_pred=y_pred, average='weighted')
    f1_samples_average = f1_score(y_true=y_true, y_pred=y_pred, average='samples')
    f1_per_class = f1_score(y_true=y_true, y_pred=y_pred, average=None)

    # In an article they propose
    optimized_f1 = np.mean(f1_per_class[np.where(f1_per_class > 0)])
    optimized_recall = np.mean(recall_per_class[np.where(recall_per_class > 0)])
    optimized_precision = np.mean(precision_per_class[np.where(precision_per_class > 0)])
    # roc_auc_micro = roc_auc_score(y_true, y_pred, average = 'micro')
    # roc_auc_macro = roc_auc_score(y_true, y_pred, average = 'macro')
    # roc_auc_weighted = roc_auc_score(y_true, y_pred, average = 'weighted')
    # roc_auc_samples = roc_auc_score(y_true, y_pred, average = 'samples')

    accuracy = accuracy_score(y_true, y_pred)

    report=classification_report(y_true,y_pred)

    # return a dictionary with all metrics
    metrics = {'precision_micro': precision_micro_average,
               'precision_macro': precision_macro_average,
               'optimized_macro_precision': optimized_precision,
               'precision_weighted': precision_weighted_average,
               'precision_samples': precision_samples_average,


               'recall_micro': recall_micro_average,
               'recall_macro': recall_macro_average,
               'optimized_macro_recall': optimized_recall,

               'recall_weighted': recall_weighted_average,
               'recall_samples': recall_samples_average,
               'f1_micro': f1_micro_average,
               'f1_macro': f1_macro_average,
               'f1_weighted': f1_weighted_average,
               'f1_samples': f1_samples_average,
               'optimized_macro_f1': optimized_f1,
               #
               # 'roc_auc_micro': roc_auc_micro,
               #  'roc_auc_macro': roc_auc_macro,
               #  'roc_auc_weighted': roc_auc_weighted,
               #  'roc_auc_samples': roc_auc_samples,

               'accuracy': accuracy,
               'report': report}
    return metrics

def compute_metrics(p: EvalPrediction):
    """
    Compute metrics for the given predictions and labels
    :param p: transformers EvalPrediction object
    :return: a dictionary of metrics
    """
    preds = p.predictions[0] if isinstance(p.predictions,
        tuple) else p.predictions
    result = multi_label_metrics(
        predictions=preds,
        labels=p.label_ids)
    print(result)
    return result

