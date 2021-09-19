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
import utils

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

    return np.mean(entropy), selected_number_sensors, accuracy, posteriors, lables, expected_posterior_entropy[selected_number_sensors-1]


def evaluate_classifiers():

    uncalibrated_classifiers, sigmoid_calibrated_classifiers, isotonic_calibrated_classifiers = get_classifiers()
    _, test_df, feature_names = expdataset.get_dataset(n_sensors=5)



    rows = []
    for goal_entropy in np.linspace(0.1, 0.65, num=100):
        entropy_inv = utils.InverseEntropy()
        goal_accuracy = entropy_inv.inverse_entropy(goal_entropy)
        print(f'Goal accuracy: {goal_accuracy}, Goal entropy: {goal_entropy}')
        df = test_df.copy()
        uncalibrated_mean_entropy, uncalibrated_selected_number_sensors, uncalibrated_accuracy, _, _, uncalibrated_expected_posterior_entropy = evaluate_classifier(uncalibrated_classifiers, goal_entropy, df)
        df = test_df.copy()
        sigmoid_mean_entropy, sigmoid_selected_number_sensors, sigmoid_accuracy, _, _, sigmoid_expected_posterior_entropy = evaluate_classifier(sigmoid_calibrated_classifiers, goal_entropy, df)
        df = test_df.copy()
        isotonic_mean_entropy, isotonic_selected_number_sensors, isotonic_accuracy, _, _, isotonic_expected_posterior_entropy = evaluate_classifier(isotonic_calibrated_classifiers, goal_entropy, df)
        
        
        rows.append({'Quantity': 'Realized accuracy', 'Calibration': 'Isotonic regression', 'Selected number of sensors':isotonic_selected_number_sensors,'Accuracy':isotonic_accuracy, 'Mean posterior entropy': isotonic_mean_entropy, 'Goal accuracy': goal_accuracy })
        rows.append({'Quantity': 'Realized accuracy', 'Calibration': 'Logistic regression', 'Selected number of sensors':sigmoid_selected_number_sensors,'Accuracy':sigmoid_accuracy, 'Mean posterior entropy': sigmoid_mean_entropy, 'Goal accuracy': goal_accuracy })
        rows.append({'Quantity': 'Realized accuracy', 'Calibration': 'Uncalibrated', 'Selected number of sensors':uncalibrated_selected_number_sensors,'Accuracy':uncalibrated_accuracy, 'Mean posterior entropy': uncalibrated_mean_entropy, 'Goal accuracy': goal_accuracy })

        rows.append({'Quantity': 'Expected accuracy', 'Calibration': 'Isotonic regression', 'Selected number of sensors':isotonic_selected_number_sensors,'Accuracy':entropy_inv.inverse_entropy(isotonic_expected_posterior_entropy), 'Mean posterior entropy': isotonic_mean_entropy, 'Goal accuracy': goal_accuracy })
        rows.append({'Quantity': 'Expected accuracy', 'Calibration': 'Logistic regression', 'Selected number of sensors':sigmoid_selected_number_sensors,'Accuracy':entropy_inv.inverse_entropy(sigmoid_expected_posterior_entropy), 'Mean posterior entropy': sigmoid_mean_entropy, 'Goal accuracy': goal_accuracy })
        rows.append({'Quantity': 'Expected accuracy', 'Calibration': 'Uncalibrated', 'Selected number of sensors':uncalibrated_selected_number_sensors,'Accuracy':entropy_inv.inverse_entropy(uncalibrated_expected_posterior_entropy), 'Mean posterior entropy': uncalibrated_mean_entropy, 'Goal accuracy': goal_accuracy })

        rows.append({'Quantity': 'Predicted accuracy', 'Calibration': 'Isotonic regression', 'Selected number of sensors':isotonic_selected_number_sensors,'Accuracy':entropy_inv.inverse_entropy(isotonic_mean_entropy), 'Mean posterior entropy': isotonic_mean_entropy, 'Goal accuracy': goal_accuracy })
        rows.append({'Quantity': 'Predicted accuracy', 'Calibration': 'Logistic regression', 'Selected number of sensors':sigmoid_selected_number_sensors,'Accuracy':entropy_inv.inverse_entropy(sigmoid_mean_entropy), 'Mean posterior entropy': sigmoid_mean_entropy, 'Goal accuracy': goal_accuracy })
        rows.append({'Quantity': 'Predicted accuracy', 'Calibration': 'Uncalibrated', 'Selected number of sensors':uncalibrated_selected_number_sensors,'Accuracy':entropy_inv.inverse_entropy(uncalibrated_mean_entropy), 'Mean posterior entropy': uncalibrated_mean_entropy, 'Goal accuracy': goal_accuracy })

    df = pd.DataFrame(rows)

    sns.lineplot(data=df, x = 'Goal accuracy', y='Accuracy', markers=False, hue='Calibration', style="Quantity")
    plt.plot([0, 1], [0, 1], "k:", label="Ideal calibration")
    plt.show()

    print('s')



if __name__=='__main__':
    evaluate_classifiers()
    # render_reliability()