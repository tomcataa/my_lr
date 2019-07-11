import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression


def preprocess(sentence):
    return " ".join(list(sentence))


class MyLR(LogisticRegression):
    def predict_proba(self, X):
        if not hasattr(self, "coef_"):
            raise NotFittedError("Call fit before prediction")

        ovr = (self.multi_class in ["ovr", "warn"] or
               (self.multi_class == 'auto' and (self.classes_.size <= 2 or
                                                self.solver == 'liblinear')))
        if ovr:
            prob = super(LogisticRegression, self).decision_function(X)
            prob *= -1
            np.exp(prob, prob)
            prob += 1
            np.reciprocal(prob, prob)
            if prob.ndim == 1:
                return np.vstack([1 - prob, prob]).T
            return prob
        else:
            decision = self.decision_function(X)
            if decision.ndim == 1:
                # Workaround for multi_class="multinomial" and binary outcomes
                # which requires softmax prediction with only a 1D decision.
                decision_2d = np.c_[-decision, decision]
            else:
                decision_2d = decision
            return softmax(decision_2d, copy=False)


le = LabelEncoder()
vect = CountVectorizer(token_pattern="(?u)\\b\\w+\\b", binary=True, ngram_range=(1, 2))
clf = MyLR(C=10, random_state=66)

df["token"] = df["sentence"].map(lambda s: preprocess(s))
X = vect.fit_transform(df["token"])
Y = le.fit_transform(df["tag"])
clf.fit(X, Y)

token = preprocess("年费收费多少")
x = vect.transform([token])
p = clf.predict_proba(x)[0]
for label, proba in zip(le.classes_, p):
    print(label, int(proba * 100))
