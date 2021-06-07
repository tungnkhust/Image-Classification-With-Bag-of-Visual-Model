from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report


def get_metrics(y_true, y_pred, average='marcor'):
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=average)
    recall = recall_score(y_true, y_pred, average=average)
    f1 = f1_score(y_true, y_pred, average=average)
    report = classification_report(y_true, y_pred)
    return acc, precision, recall, f1, report