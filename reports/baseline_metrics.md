# Baseline Metrics — Logistic Regression
Model artifact: `models/baseline_logreg.joblib`
## Validation
- Accuracy: 0.804
- Precision: 0.833
- Recall: 0.714
- F1: 0.769
- ROC AUC: 0.869

Confusion matrix (val):
```
[[22  3]
 [ 6 15]]
```

Classification report (val):
```
              precision    recall  f1-score   support

           0       0.79      0.88      0.83        25
           1       0.83      0.71      0.77        21

    accuracy                           0.80        46
   macro avg       0.81      0.80      0.80        46
weighted avg       0.81      0.80      0.80        46

```

## Test
- Accuracy: 0.848
- Precision: 0.792
- Recall: 0.905
- F1: 0.844
- ROC AUC: 0.939

Confusion matrix (test):
```
[[20  5]
 [ 2 19]]
```

Classification report (test):
```
              precision    recall  f1-score   support

           0       0.91      0.80      0.85        25
           1       0.79      0.90      0.84        21

    accuracy                           0.85        46
   macro avg       0.85      0.85      0.85        46
weighted avg       0.86      0.85      0.85        46

```
