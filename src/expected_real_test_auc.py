import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


test_real_labels_path = "../data/test_real_labels.txt"
#test_real_predictions_path = "../data/94_test_predictions_data_augmentation.txt"
test_real_predictions_path = "../data/test_predictions_data_augmentation.txt"
#test_real_predictions_path = "../data/test_predictions_extracted_features.txt"


label_names = ["names", "labels"]
predictions_names = ["names", "predictions"]
label_df = pd.read_csv(test_real_labels_path, header=None, names=label_names)
predictions_df = pd.read_csv(test_real_predictions_path, header=None, names=predictions_names)
final_df = pd.merge(label_df, predictions_df, on='names')
final_df.dropna(inplace=True)

labels = final_df["labels"].to_numpy()
predictions = final_df["predictions"].to_numpy()
score = metrics.roc_auc_score(labels, predictions)
print(score)

final_df.sort_values(by=["predictions"], ascending=False, inplace=True)
print(final_df.to_string())


fpr, tpr, _ = roc_curve(labels, predictions)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC curve (area = {score:.3f})")
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()