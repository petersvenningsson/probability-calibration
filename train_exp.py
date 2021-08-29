from math import e
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import accuracy_score
from sklearn.calibration import calibration_curve

import classifiers
import expdataset

n_classes = 2
max_sensors = 5


def render_reliability():
    _, test_df, feature_names = expdataset.get_dataset(n_sensors=5)
    test_df = test_df.groupby('lable', group_keys=False).apply(lambda x: x.sample(len(test_df), replace=True)) # resample to uniform class distribution

    uncalibrated_classifiers, sigmoid_calibrated_classifiers, isotonic_calibrated_classifiers = get_classifiers()
    
    goal_entropy = 0 
    _, _, _, posteriors, lables = evaluate_classifier(uncalibrated_classifiers, goal_entropy, test_df)
    fraction_of_positives, mean_predicted_value = calibration_curve(lables-1, posteriors[:,1], n_bins=8)
    plt.plot(fraction_of_positives, mean_predicted_value, "s-", label="Uncalibrated")

    _, _, _, posteriors, lables = evaluate_classifier(sigmoid_calibrated_classifiers, goal_entropy, test_df)
    fraction_of_positives, mean_predicted_value = calibration_curve(lables-1, posteriors[:,1], n_bins=8)
    plt.plot(fraction_of_positives, mean_predicted_value, "s-", label="Logistic regression")

    _, _, _, posteriors, lables = evaluate_classifier(isotonic_calibrated_classifiers, goal_entropy, test_df)
    fraction_of_positives, mean_predicted_value = calibration_curve(lables-1, posteriors[:,1], n_bins=8)
    plt.plot(fraction_of_positives, mean_predicted_value, "s-", label="Isotonic regression")

    plt.plot([0, 1], [0, 1], "k:", label="Ideal calibration")
    plt.ylabel("Fraction of positives")
    plt.xlabel("Predicted confidence")
    plt.ylim([-0.05, 1.05])
    plt.legend(loc="lower right")

    plt.show()
    print('s')


def get_classifiers():
    uninformative_prior = np.array([[1/n_classes for _ in range(n_classes)],]).T

    uncalibrated_classifiers = {}
    isotonic_calibrated_classifiers = {}
    sigmoid_calibrated_classifiers = {}

    for n_sensors in range(1, max_sensors+1):
        dataset, _, feature_names = expdataset.get_dataset(n_sensors = n_sensors)

        uncalibrated_classifier = classifiers.ClassifierComposition(feature_names, discriminant_model='Gaussian')
        isotonic_calibrated_classifier = classifiers.ClassifierComposition(feature_names, discriminant_model='isotonic_calibration')
        sigmoid_calibrated_classifier = classifiers.ClassifierComposition(feature_names, discriminant_model='sigmoid_calibration')

        uncalibrated_classifier.fit(dataset.copy())
        isotonic_calibrated_classifier.fit(dataset.copy())
        sigmoid_calibrated_classifier.fit(dataset.copy())

        uncalibrated_information = uncalibrated_classifier.information(uninformative_prior, n=100000)
        isotonic_calibrated_information = isotonic_calibrated_classifier.information(uninformative_prior, n=100000)
        sigmoid_calibrated_information = sigmoid_calibrated_classifier.information(uninformative_prior, n=100000)


        # Classifiers are stored as (classifier, information) tuples
        uncalibrated_classifiers[n_sensors] = (uncalibrated_classifier, uncalibrated_information)
        sigmoid_calibrated_classifiers[n_sensors] = (sigmoid_calibrated_classifier, sigmoid_calibrated_information)
        isotonic_calibrated_classifiers[n_sensors] = (isotonic_calibrated_classifier, isotonic_calibrated_information)

    return uncalibrated_classifiers, sigmoid_calibrated_classifiers, isotonic_calibrated_classifiers


    
def evaluate_classifier(classifiers, goal_entropy, test_df):
    initial_entropy = np.log(n_classes)

    # Decide how many sensors should be used
    information = [classifiers[i][1] for i in range(1, max_sensors+1)]

    expected_posterior_entropy = initial_entropy - information
    selected_number_sensors = [i for i in range(1,max_sensors+1) if expected_posterior_entropy[i-1]<goal_entropy]
    if selected_number_sensors:
        selected_number_sensors = selected_number_sensors[0]
    else:
        selected_number_sensors = max_sensors
    
    classifier = classifiers[selected_number_sensors][0]
    # Evaluate on test dataset

    posteriors = classifier.predict_proba(test_df)

    entropy = stats.entropy(posteriors, base=e, axis=1)

    # Evaluate accuracy on test set
    predictions = posteriors.argmax(axis=1)
    lables = test_df['lable'].to_numpy()
    accuracy = accuracy_score(predictions + 1, lables)

    return np.mean(entropy), selected_number_sensors, accuracy, posteriors, lables


def evaluate_classifiers():
    goal_entropy = 0.1 # 2 vad bra
    uncalibrated_classifiers, sigmoid_calibrated_classifiers, isotonic_calibrated_classifiers = get_classifiers()
    _, test_df, feature_names = expdataset.get_dataset(n_sensors=5)

    df = test_df.copy()
    uncalibrated_mean_entropy, uncalibrated_selected_number_sensors, uncalibrated_accuracy, _, _ = evaluate_classifier(uncalibrated_classifiers, goal_entropy, df)
    df = test_df.copy()
    sigmoid_mean_entropy, sigmoid_selected_number_sensors, sigmoid_accuracy,_,_ = evaluate_classifier(sigmoid_calibrated_classifiers, goal_entropy, df)
    df = test_df.copy()
    isotonic_mean_entropy, isotonic_selected_number_sensors, isotonic_accuracy,_,_ = evaluate_classifier(isotonic_calibrated_classifiers, goal_entropy, df)

    print('s')



if __name__=='__main__':
    # evaluate_classifiers()
    render_reliability()