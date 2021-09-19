import numpy as np
from sklearn.naive_bayes import GaussianNB
from scipy.special import logsumexp
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder
from scipy import stats
from math import e

import pandas as pd

import utils

class_set = 2
entropy_inv = utils.InverseEntropy()

class ClassifierComposition:
    """ A composition of a generative model and a discriminant model
    """
    def __init__(self, features, discriminant_model):
        self.priors = [1/class_set for _ in range(class_set)]
        self.discriminant_model = discriminant_model
        self.features = features

        if discriminant_model == 'Gaussian':
            self.model = GaussianNaiveBayes(self.features, self.priors)
            self.generative_model = self.model

        elif discriminant_model == 'isotonic_calibration':
            self.model = GaussianNaiveBayes(self.features, self.priors, calibrated='isotonic')
            self.generative_model = GaussianNaiveBayes(self.features, self.priors)

        elif discriminant_model == 'sigmoid_calibration':
            self.model = GaussianNaiveBayes(self.features, self.priors, calibrated='sigmoid')
            self.generative_model = GaussianNaiveBayes(self.features, self.priors)

        elif discriminant_model == 'logistic':
            self.model = LogisticReg(self.features)
            self.generative_model = GaussianNaiveBayes(self.features, self.priors)

        else:
            raise ValueError()


    def fit(self, df):
        self.training_df = df
        self.model.fit(df)
        self.generative_model.fit(df)


    def predict_proba(self, df):
        return self.model.predict_proba(df)


    def predict(self, df):
        return self.model.predict(df)


    def _information(self, belief, n=1000, espilon = 0.00001):

        # If Gaussian model then the generative model can be sampled. Else the training dataset is sampled.
        if self.discriminant_model == 'Gaussian':
            theta_ = self.generative_model.model.theta_
            sigma_ = self.generative_model.model.sigma_

            class_sample = np.random.choice(list(range(class_set)), n, p=belief.squeeze())

            means = np.array(list(map(lambda x: theta_[x,:], class_sample)))
            std = np.sqrt(np.array(list(map(lambda x: sigma_[x,:], class_sample))))

            X = np.random.standard_normal((n, len(self.features)))
            X = X*std + means

            rows = dict(zip(self.features, X.T))
            rows['lable'] = class_sample
            sample_probability = np.log(self.predict_proba(pd.DataFrame(rows)) + espilon)

            enumerator = np.take_along_axis(sample_probability, class_sample[:,None], axis=1).squeeze()
            denomonator = logsumexp((sample_probability + np.repeat(np.log(belief), n, axis=1).T), axis=1).squeeze()

            _information = enumerator - denomonator

        else:
            samples = self.training_df.groupby('lable', group_keys=False).apply(lambda x: x.sample(int(n/belief.size), replace=True))

            sample_probability = np.log(self.predict_proba(samples[self.features]) + espilon)

            enumerator = np.take_along_axis(sample_probability, (samples['lable'].to_numpy().astype(int))[:,None], axis=1).squeeze()
            denomonator = logsumexp((sample_probability + np.repeat(np.log(belief), n, axis=1).T), axis=1).squeeze()

            _information = enumerator - denomonator

        return _information


    def expected_confidence(self, belief, n=1000):
        samples = self.training_df.groupby('lable', group_keys=False).apply(lambda x: x.sample(int(n/belief.size), replace=True))
        sample_probability = self.predict_proba(samples[self.features])
        mean_confidence = np.mean(np.max(sample_probability, axis=1))
        return mean_confidence



    def information(self, args, **kwargs):
        return np.mean(self._information(*args, **kwargs))


    def expected_accuracy(self, belief, n=1000, espilon = 0.00001):
        initial_entropy = stats.entropy(belief, base=e)
        information_samples = self._information(belief, n, espilon)
        posterior_entropy_sampels = initial_entropy - information_samples

        confidences = entropy_inv.inverse_entropy(posterior_entropy_sampels)

        return np.mean(confidences)


class LogisticReg(LogisticRegression):
    def __init__(self, features):
        super().__init__(class_weight='balanced')
        self.scaler = StandardScaler()
        self.features = features

    def fit(self, df):

        X = df[self.features].to_numpy()
        y = df['lable'].to_numpy()
        X = self.scaler.fit_transform(X)
        return super().fit(X, y)

    def predict(self, df):
        X = df[self.features].to_numpy()
        y = df['lable'].to_numpy()
        X = self.scaler.transform(X)
        return super().predict(X, y)

    def predict_proba(self, df):
        X = df[self.features].to_numpy()
        X = self.scaler.transform(X)
        return super().predict_proba(X)

    def predict_log_proba(self, df):
        X = df[self.features].to_numpy()
        X = self.scaler.transform(X)
        return super().predict_log_proba(X)


class GaussianNaiveBayes():
    def __init__(self, features, priors, calibrated=False):
        self.features = features
        self.priors = priors
        self.calibrated = calibrated

        if calibrated:
            self.model = CalibratedClassifierCV(base_estimator=GaussianNB(priors=self.priors, var_smoothing=1e-6), method=calibrated)
        else:
            self.model = GaussianNB(priors=self.priors, var_smoothing = 1e-6)


    def fit(self, df):

        if self.calibrated:
            # Resample dataset to uniform class distribution
            df = df.groupby('lable', group_keys=False).apply(lambda x: x.sample(len(df), replace=True))
            # GroupShuffleSplit is used to avoid information leak between splits caused by several copies of one sample
            cv = GroupShuffleSplit(n_splits=5).split(df[self.features].to_numpy(), y=df['lable'].to_numpy(), groups=df.index.to_numpy())
            self.model.cv = cv
            
        self.model.fit(df[self.features].to_numpy(), df['lable'].to_numpy())

    def predict(self, df):
        return self.model.predict(df[self.features].to_numpy())

    def predict_proba(self, df):
       return self.model.predict_proba(df[self.features].to_numpy())

    def predict_log_proba(self, df):
        return self.model.predict_log_proba(df[self.features].to_numpy())