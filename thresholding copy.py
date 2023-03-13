
import json
import numpy as np
import math


# TPR VS FPR
def roc_curve(y_true, y_prob, thresholds):

    fpr = []
    tpr = []

    for threshold in thresholds:

        y_pred = np.where(y_prob >= threshold, 1, 0)

        fp = np.sum((y_pred == 1) & (y_true == 0))
        tp = np.sum((y_pred == 1) & (y_true == 1))

        fn = np.sum((y_pred == 0) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))

        fpr.append(fp / (fp + tn))
        tpr.append(tp / (tp + fn))

    return [fpr, tpr]


def get_data(polarizingqs_path, triviaQAmini_path, triviaQA_path=None):
    # triviaQAmini_path
    with open(triviaQAmini_path, 'r') as f:
        data = json.load(f)
    
    questions = data.keys()
    trivia_offdiaggap = []
    for question in questions:
        if not math.isnan(data[question]['off_diagonal_gap']):
            trivia_offdiaggap.append(data[question]['off_diagonal_gap'])

    # triviaQA_path
    if triviaQA_path: 
        with open(triviaQA_path, 'r') as f:
            data = json.load(f)

        questions = data.keys()
        for question in questions:
            if not math.isnan(data[question]['off_diagonal_gap']):
                trivia_offdiaggap.append(data[question]['off_diagonal_gap'])

    # polarizingqs_path
    polarizing_offdiaggap = []
    polarizing_offdiaggap_dict = {}

    with open(polarizingqs_path, 'r') as f:
        data = json.load(f)
    questions = data.keys()
    for question in questions:
        if question != 'avg':
            if not math.isnan(data[question]['off_diagonal_gap']):
                polarizing_offdiaggap.append(data[question]['off_diagonal_gap'])
                polarizing_offdiaggap_dict[question]=data[question]['off_diagonal_gap']
    
    return trivia_offdiaggap, polarizing_offdiaggap, polarizing_offdiaggap_dict

def optimal_threshold(polarizingqs_path, triviaQAmini_path, triviaQA_path=None):

    trivia_offdiaggap, polarizing_offdiaggap, polarizing_offdiaggap_dict  = get_data(polarizingqs_path, triviaQAmini_path, triviaQA_path)
    thresholds = []
    for i in np.linspace(0, 49, 50):
        thresholds.append(np.percentile(polarizing_offdiaggap, i))

    # y_true is a 0 or 1 label where 1 is polarizing question and 0 is nonpolarizing question (triviaQA) 
    y_true = np.concatenate((np.ones(len(polarizing_offdiaggap)), np.zeros(len(trivia_offdiaggap))), axis=0)
    # y_prob is the agreement scores, append these together, first with polarizing agreement scores then nonpolarizing agreement scores
    y_prob = np.concatenate((polarizing_offdiaggap,trivia_offdiaggap), axis=0)
    [fpr, tpr] = roc_curve(y_true, y_prob, thresholds)
    fpr, tpr = np.array(fpr), np.array(tpr)
    # PICK THRESHOLD THAT MAXIMIZES TPR-FPR
    optimal_threshold_percentile, optimal_threshold= np.linspace(0, 49, 50)[np.argmax((tpr-fpr))], thresholds[np.argmax((tpr-fpr))]
    return optimal_threshold, trivia_offdiaggap, polarizing_offdiaggap, polarizing_offdiaggap_dict
