Hold-out test set (`test_data.csv`): **114** tumours (42 malignant, 72 benign), never used for training or tuning. Models were trained on **455** tumours, with hyperparameters chosen by 5-fold stratified cross-validation on the training set (scoring: MCC). Precision, recall and F1 treat **Malignant** as the positive class; the decision threshold is 0.5.

### Comparison table (test data)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9825 | 0.9964 | 0.9762 | 0.9762 | 0.9762 | 0.9623 |
| Decision Tree | 0.9649 | 0.9714 | 0.9524 | 0.9524 | 0.9524 | 0.9246 |
| KNN | 0.9825 | 0.9959 | 0.9762 | 0.9762 | 0.9762 | 0.9623 |
| Naive Bayes | 0.9561 | 0.9897 | 0.9302 | 0.9524 | 0.9412 | 0.9064 |
| Support Vector Machine | 0.9737 | 0.9987 | 0.9535 | 0.9762 | 0.9647 | 0.9439 |

### Tuned hyperparameters

| ML Model Name | Selected hyperparameters | Cross-validated MCC (mean ± sd) |
|---|---|---:|
| Logistic Regression | C=1 | 0.9405 ± 0.0111 |
| Decision Tree | criterion=entropy, max_depth=5, min_samples_leaf=5 | 0.8828 ± 0.0595 |
| KNN | n_neighbors=7, weights=uniform | 0.9301 ± 0.0330 |
| Naive Bayes | var_smoothing=1e-09 | 0.8494 ± 0.0509 |
| Support Vector Machine | C=10, gamma=0.01 | 0.9539 ± 0.0251 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Once the features are standardised the two classes are close to linearly separable, so a regularised linear boundary fits well. L2 regularisation keeps the weights stable even though radius, perimeter and area are near copies of each other (correlation up to 0.998). On the test set: accuracy 0.982, AUC 0.996, MCC 0.962 (rank 1 of 5 by MCC); it missed 1 of 42 malignant tumours and raised 1 false alarm. |
| Decision Tree | Splits on one feature at a time, so it approximates the diagonal class boundary with axis-aligned steps. It is the easiest model to explain, but it is the most sensitive to which tumours land in the training set and its probabilities are coarse leaf proportions. On the test set: accuracy 0.965, AUC 0.971, MCC 0.925 (rank 4 of 5 by MCC); it missed 2 of 42 malignant tumours and raised 2 false alarms. |
| KNN | Works well after standardisation because malignant tumours occupy a compact region of large, irregular nuclei. The many correlated size features give tumour size extra weight in the distance, and its probabilities are coarse neighbour vote shares. On the test set: accuracy 0.982, AUC 0.996, MCC 0.962 (rank 1 of 5 by MCC); it missed 1 of 42 malignant tumours and raised 1 false alarm. |
| Naive Bayes | Gaussian Naive Bayes assumes features are independent given the class, but many features here are near duplicates, so the same evidence is counted several times and its probabilities are over-confident. This usually hurts its hard predictions more than its ranking (AUC). On the test set: accuracy 0.956, AUC 0.990, MCC 0.906 (rank 5 of 5 by MCC); it missed 2 of 42 malignant tumours and raised 3 false alarms. |
| Support Vector Machine | The RBF kernel can bend the decision boundary, but because the data is nearly linear it behaves much like Logistic Regression. Its probabilities come from Platt scaling fitted inside the SVC. On the test set: accuracy 0.974, AUC 0.999, MCC 0.944 (rank 3 of 5 by MCC); it missed 1 of 42 malignant tumours and raised 2 false alarms. |

**Overall winner: Logistic Regression** - highest test MCC (0.9623) with accuracy 0.9825, AUC 0.9964 and F1 0.9762 (cross-validated MCC 0.9405). KNN ties exactly on test MCC; Logistic Regression is placed first on the tie-break (higher AUC, then cross-validated MCC). Note that Support Vector Machine has the highest cross-validated MCC (0.9539, averaged over 5 folds of the training data) and the highest test AUC (0.9987), ranking 3 on test MCC. With only 114 test tumours one misclassification moves MCC by about 0.02, so it is an equally strong choice. MCC is used for the ranking because it uses all four cells of the confusion matrix and is not flattered by the 63/37 class imbalance.
