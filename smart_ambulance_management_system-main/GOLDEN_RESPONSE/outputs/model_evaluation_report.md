# Model Evaluation Report: Intelligent Emergency Response Optimization

This report summarizes the performance evaluation and model comparisons for the emergency response predictive models.

## Target Definitions & Selections

| Target Variable | Type | Best Model Selected | Test Accuracy | F1-Score |
|---|---|---|---|---|
| `is_delayed` | Binary Classification | **gradient_boosting** | 0.9848 | 0.9882 |
| `optimal_zone` | Multi-class Classification | **random_forest** | 0.0101 | 0.0039 |
| `arrival_category` | Multi-class Classification | **gradient_boosting** | 0.9747 | 0.9705 |

## Detailed Performance Comparison

### Target: `is_delayed`

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Random forest | 0.9444 | 0.9259 | 0.9921 | 0.9579 | 0.9921 |
| Gradient boosting | 0.9848 | 0.9767 | 1.0000 | 0.9882 | 0.9886 |

#### Confusion Matrices

**Random forest Confusion Matrix:**
```
[[ 62  10]
 [  1 125]]
```

**Gradient boosting Confusion Matrix:**
```
[[ 69   3]
 [  0 126]]
```

### Target: `optimal_zone`

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Random forest | 0.0101 | 0.0034 | 0.0049 | 0.0039 | 0.5000 |
| Gradient boosting | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.5000 |

#### Confusion Matrices

**Random forest Confusion Matrix:**
```
[[0 0 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]]
```

**Gradient boosting Confusion Matrix:**
```
[[0 0 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]]
```

### Target: `arrival_category`

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Random forest | 0.9242 | 0.9518 | 0.8288 | 0.8671 | 0.9851 |
| Gradient boosting | 0.9747 | 0.9610 | 0.9812 | 0.9705 | 0.9978 |

#### Confusion Matrices

**Random forest Confusion Matrix:**
```
[[12 10  0]
 [ 0 97  2]
 [ 0  3 74]]
```

**Gradient boosting Confusion Matrix:**
```
[[22  0  0]
 [ 2 96  1]
 [ 0  2 75]]
```
