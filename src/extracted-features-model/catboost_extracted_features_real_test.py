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
from catboost import CatBoostClassifier, Pool

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

        weights = []
        for i in range(len(targets)):
            if(targets[i] == 0.0):
                weights.append(0.18)
            else:
                weights.append(0.82)


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



df_train = pd.read_csv("../../data/embedded-datasets/train_embedded_dataset.csv")
df_test = pd.read_csv("../../data/embedded-datasets/test_embedded_dataset.csv")
features = (["v{}".format(str(x)) for x in range(0, 512)] 
          + ["h{}".format(str(x)) for x in range(10, 14)])


X = df_train.loc[:][features].to_numpy()
y = df_train["label"].to_numpy()
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)

X_test = df_test.loc[:][features]
test_files = df_test.loc[:]["file"]


train_pool = Pool(X_train, y_train)
validate_pool = Pool(X_val, y_val)

model = CatBoostClassifier(loss_function=FocalLossObjective(), 
						   eval_metric="AUC", 
						   iterations=500,
					#	   verbose=False
						   )

model.fit(train_pool, eval_set=validate_pool)

test_predictions = model.predict_proba(X_test)[:, 1]

df_predictions = pd.DataFrame({"file":test_files, "label":test_predictions})
df_predictions.to_csv("../../data/test_predictions_extracted_features.txt", header=False, index=False)