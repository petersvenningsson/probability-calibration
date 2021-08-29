from math import e
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import simdataset
import classifiers
from sklearn.metrics import accuracy_score
from sklearn.calibration import calibration_curve

n_classes = 2


max_sensors = 50


def render_reliability():
    sensor_correlation = 0.2
    uncalibrated_classifiers, sigmoid_calibrated_classifiers, isotonic_calibrated_classifiers = get_classifiers(sensor_correlation = sensor_correlation)
    
    _, _, _, posteriors, lables = evaluate_classifier(uncalibrated_classifiers, sensor_correlation, goal_entropy = 0.5)
    fraction_of_positives, mean_predicted_value = calibration_curve(lables, posteriors[:,1], n_bins=8)
    plt.plot(fraction_of_positives, mean_predicted_value, "s-", label="Uncalibrated")

    _, _, _, posteriors, lables = evaluate_classifier(sigmoid_calibrated_classifiers, sensor_correlation, goal_entropy = 0.5)
    fraction_of_positives, mean_predicted_value = calibration_curve(lables, posteriors[:,1], n_bins=8)
    plt.plot(fraction_of_positives, mean_predicted_value, "s-", label="Logistic regression")

    _, _, _, posteriors, lables = evaluate_classifier(isotonic_calibrated_classifiers, sensor_correlation, goal_entropy = 0.5)
    fraction_of_positives, mean_predicted_value = calibration_curve(lables, posteriors[:,1], n_bins=8)
    plt.plot(fraction_of_positives, mean_predicted_value, "s-", label="Isotonic regression")

    plt.plot([0, 1], [0, 1], "k:", label="Ideal calibration")
    plt.ylabel("Fraction of positives")
    plt.xlabel("Predicted confidence")
    plt.ylim([-0.05, 1.05])
    plt.legend(loc="lower right")

    plt.show()


def render_information():
    uninformative_prior = np.array([[1/n_classes for _ in range(n_classes)],]).T

    rows = []
    for sensor_correlation in np.linspace(0, 0.6, num=10):
        dataset, feature_names = simdataset.get_dataset(sensor_correlation = sensor_correlation, n_sensors = 10, n_total_samples=1000000)

        uncalibrated_classifier = classifiers.ClassifierComposition(feature_names, discriminant_model='Gaussian')
        isotonic_calibrated_classifier = classifiers.ClassifierComposition(feature_names, discriminant_model='isotonic_calibration')
        sigmoid_calibrated_classifier = classifiers.ClassifierComposition(feature_names, discriminant_model='sigmoid_calibration')

        uncalibrated_classifier.fit(dataset)
        isotonic_calibrated_classifier.fit(dataset)
        sigmoid_calibrated_classifier.fit(dataset)

        isotonic_calibrated_information = isotonic_calibrated_classifier.information(uninformative_prior, n=10000)
        sigmoid_calibrated_information = sigmoid_calibrated_classifier.information(uninformative_prior, n=10000)
        uncalibrated_information = uncalibrated_classifier.information(uninformative_prior, n=10000)

        rows.append({ 'Calibration': 'Isotonic regression', 'Information': isotonic_calibrated_information, 'Sensor correlation': sensor_correlation })
        rows.append({ 'Calibration': 'Logistic regression', 'Information': sigmoid_calibrated_information, 'Sensor correlation': sensor_correlation })
        rows.append({ 'Calibration': 'Uncalibrated', 'Information': uncalibrated_information, 'Sensor correlation': sensor_correlation })
    df = pd.DataFrame(rows)

    sns.lineplot(data=df, x = 'Sensor correlation', y='Information', markers=True, dashes=False, hue='Calibration', style="Calibration")
    plt.show()
    print('s')



def get_classifiers(sensor_correlation = 0.4):
    uninformative_prior = np.array([[1/n_classes for _ in range(n_classes)],]).T

    uncalibrated_classifiers = {}
    isotonic_calibrated_classifiers = {}
    sigmoid_calibrated_classifiers = {}

    for n_sensors in range(1, max_sensors+1):
        dataset, feature_names = simdataset.get_dataset(sensor_correlation = sensor_correlation, n_sensors = n_sensors, n_total_samples=10000)

        uncalibrated_classifier = classifiers.ClassifierComposition(feature_names, discriminant_model='Gaussian')
        isotonic_calibrated_classifier = classifiers.ClassifierComposition(feature_names, discriminant_model='isotonic_calibration')
        sigmoid_calibrated_classifier = classifiers.ClassifierComposition(feature_names, discriminant_model='sigmoid_calibration')

        uncalibrated_classifier.fit(dataset)
        isotonic_calibrated_classifier.fit(dataset)
        sigmoid_calibrated_classifier.fit(dataset)

        isotonic_calibrated_information = isotonic_calibrated_classifier.information(uninformative_prior, n=1000)
        sigmoid_calibrated_information = sigmoid_calibrated_classifier.information(uninformative_prior, n=1000)
        uncalibrated_information = uncalibrated_classifier.information(uninformative_prior, n=1000)

        # Classifiers are stored as (classifier, information) tuples
        uncalibrated_classifiers[n_sensors] = (uncalibrated_classifier, uncalibrated_information)
        sigmoid_calibrated_classifiers[n_sensors] = (sigmoid_calibrated_classifier, sigmoid_calibrated_information)
        isotonic_calibrated_classifiers[n_sensors] = (isotonic_calibrated_classifier, isotonic_calibrated_information)

    return uncalibrated_classifiers, sigmoid_calibrated_classifiers, isotonic_calibrated_classifiers


def evaluate_classifier(classifiers, sensor_correlation, goal_entropy):
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

    dataset, _ = simdataset.get_dataset(sensor_correlation = sensor_correlation, n_sensors = selected_number_sensors, n_total_samples=10000)
    posteriors = classifier.predict_proba(dataset)

    entropy = stats.entropy(posteriors, base=e, axis=1)

    # Evaluate accuracy on test set
    predictions = posteriors.argmax(axis=1)
    lables = dataset['lable'].to_numpy()
    accuracy = accuracy_score(predictions, dataset['lable'].to_numpy())

    return np.mean(entropy), selected_number_sensors, accuracy, posteriors, lables


def evaluate_n_sensors():
    rows = []
    for sensor_correlation in np.linspace(0, 0.6, num=10):

        print(sensor_correlation)
        uncalibrated_classifiers, sigmoid_calibrated_classifiers, isotonic_calibrated_classifiers = get_classifiers(sensor_correlation = sensor_correlation)

        classifiers = uncalibrated_classifiers
        for selected_number_sensors in range(1,max_sensors+1):
            classifier = classifiers[selected_number_sensors][0]
            information = classifiers[selected_number_sensors][1]
            dataset, _ = simdataset.get_dataset(sensor_correlation = sensor_correlation, n_sensors = selected_number_sensors, n_total_samples=10000)
            posteriors = classifier.predict_proba(dataset)

            entropy = stats.entropy(posteriors, base=e, axis=1)

            # Evaluate accuracy on test set
            predictions = posteriors.argmax(axis=1)
            accuracy = accuracy_score(predictions, dataset['lable'].to_numpy())

            rows.append({ 'selected_number_sensors': selected_number_sensors,'Accuracy':accuracy, 'information': information, 'Mean posterior entropy': np.mean(entropy), 'Sensor correlation': sensor_correlation })
    
    df = pd.DataFrame(rows)

    sns.lineplot(data=df, x = 'Sensor correlation', y='Mean posterior entropy', markers=True, dashes=False, hue='selected_number_sensors', style="selected_number_sensors")
    plt.show()

    sns.lineplot(data=df, x = 'Sensor correlation', y='information', markers=True, dashes=False, hue='selected_number_sensors', style="selected_number_sensors")
    plt.show()

    sns.lineplot(data=df, x = 'Sensor correlation', y='Accuracy', markers=True, dashes=False, hue='selected_number_sensors', style="selected_number_sensors")
    plt.show()
    print('s')


def evaluate_classifiers():
    goal_entropy = 0.5 # maximum entropy is log(2) approx 0.7.

    rows = []
    for sensor_correlation in np.linspace(0, 0.6, num=10):

        print(sensor_correlation)
        uncalibrated_classifiers, sigmoid_calibrated_classifiers, isotonic_calibrated_classifiers = get_classifiers(sensor_correlation = sensor_correlation)

        uncalibrated_mean_entropy, uncalibrated_selected_number_sensors, uncalibrated_accuracy, _, _ = evaluate_classifier(uncalibrated_classifiers, sensor_correlation, goal_entropy)
        sigmoid_mean_entropy, sigmoid_selected_number_sensors, sigmoid_accuracy, _, _ = evaluate_classifier(sigmoid_calibrated_classifiers, sensor_correlation, goal_entropy)
        isotonic_mean_entropy, isotonic_selected_number_sensors, isotonic_accuracy, _, _ = evaluate_classifier(isotonic_calibrated_classifiers, sensor_correlation, goal_entropy)

        rows.append({ 'Calibration': 'Isotonic regression', 'Selected number of sensors':isotonic_selected_number_sensors,'Accuracy':isotonic_accuracy, 'Mean posterior entropy': isotonic_mean_entropy, 'Sensor correlation': sensor_correlation })
        rows.append({ 'Calibration': 'Logistic regression', 'Selected number of sensors':sigmoid_selected_number_sensors,'Accuracy':sigmoid_accuracy, 'Mean posterior entropy': sigmoid_mean_entropy, 'Sensor correlation': sensor_correlation })
        rows.append({ 'Calibration': 'Uncalibrated', 'Selected number of sensors':uncalibrated_selected_number_sensors,'Accuracy':uncalibrated_accuracy, 'Mean posterior entropy': uncalibrated_mean_entropy, 'Sensor correlation': sensor_correlation })

    df = pd.DataFrame(rows)

    sns.lineplot(data=df, x = 'Sensor correlation', y='Mean posterior entropy', markers=True, dashes=False, hue='Calibration', style="Calibration")
    plt.show()

    sns.lineplot(data=df, x = 'Sensor correlation', y='Selected number of sensors', markers=True, dashes=False, hue='Calibration', style="Calibration")
    plt.show()

    sns.lineplot(data=df, x = 'Sensor correlation', y='Accuracy', markers=True, dashes=False, hue='Calibration', style="Calibration")
    plt.show()
    print('s')


if __name__=='__main__':
    # evaluate_classifiers()
    # render_reliability()
    # render_information()