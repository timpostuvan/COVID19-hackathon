import pandas as pd
import math
from six.moves import xrange
from catboost import Pool, CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

import warnings
warnings.filterwarnings('ignore')


class FocalLossObjective(object):
    def calc_ders_range(self, approxes, targets, weights):
        # approxes, targets, weights are indexed containers of floats
        # (containers with only __len__ and __getitem__ defined).
        # weights parameter can be None.
        # Returns list of pairs (der1, der2)
        gamma = 2.
        # alpha = 1.
        assert len(approxes) == len(targets)
        if weights is not None:
            assert len(weights) == len(approxes)
        
        exponents = []
        for index in xrange(len(approxes)):
            exponents.append(math.exp(approxes[index]))

        result = []
        for index in xrange(len(targets)):
            p = exponents[index] / (1 + exponents[index])

            if targets[index] > 0.0:
                der1 = -((1-p)**(gamma-1))*(gamma * math.log(p) * p + p - 1)/p
                der2 = gamma*((1-p)**gamma)*((gamma*p-1)*math.log(p)+2*(p-1))
            else:
                der1 = (p**(gamma-1)) * (gamma * math.log(1 - p) - p)/(1 - p)
                der2 = p**(gamma-2)*((p*(2*gamma*(p-1)-p))/(p-1)**2 + (gamma-1)*gamma*math.log(1 - p))

            if weights is not None:
                der1 *= weights[index]
                der2 *= weights[index]

            result.append((der1, der2))

        return result



df = pd.read_csv("../../data/embedded-datasets/train_embedded_dataset.csv")
df = df.sample(frac=1, random_state=744)


print("COVID positive cases: {}".format(len(df[df.label==1].index)))
features = ["v{}".format(str(x)) for x in range(0, 512)]


aucs = []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=744)
target = df.label
for train_index, test_index in skf.split(np.zeros(len(target)), target):
    X_train = df.loc[train_index][features]
    X_test = df.loc[test_index][features]
    
    y_train = df.loc[train_index]['label']
    y_test = df.loc[test_index]['label']
    
    #clf1 = LogisticRegression(multi_class='multinomial', random_state=1)
    #clf2 = CatBoostClassifier(loss_function=FocalLossObjective(), eval_metric="AUC", verbose=False)
    #clf3 = GaussianNB()

    #model = VotingClassifier(estimators=[('lr', clf1), ('cb', clf2), ('gnb', clf3)], voting='soft')
    model = CatBoostClassifier(loss_function=FocalLossObjective(), eval_metric="AUC", verbose=False, iterations=500)

    model.fit(X_train, y_train)
    aucs.append(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))


print(np.mean(aucs))