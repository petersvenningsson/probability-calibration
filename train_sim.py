import pickle

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

import utils
from simconfig import max_sensors, samples_information, train_dataset_samples, test_dataset_samples, goal_entropy, goal_accuracy

n_classes = 2
entropy_inv = utils.InverseEntropy()

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

        isotonic_calibrated_information = isotonic_calibrated_classifier.information(uninformative_prior, n=samples_information)
        sigmoid_calibrated_information = sigmoid_calibrated_classifier.information(uninformative_prior, n=samples_information)
        uncalibrated_information = uncalibrated_classifier.information(uninformative_prior, n=samples_information)

        rows.append({ 'Calibration': 'Isotonic regression', 'Information': isotonic_calibrated_information, 'Sensor correlation': sensor_correlation })
        rows.append({ 'Calibration': 'Logistic regression', 'Information': sigmoid_calibrated_information, 'Sensor correlation': sensor_correlation })
        rows.append({ 'Calibration': 'Uncalibrated', 'Information': uncalibrated_information, 'Sensor correlation': sensor_correlation })
    df = pd.DataFrame(rows)

    sns.lineplot(data=df, x = 'Sensor correlation', y='Information', markers=True, dashes=False, hue='Calibration', style="Calibration")
    plt.show()
    print('s')



def get_classifiers(sensor_correlation = 0.4, feature_correlation = 0):
    uninformative_prior = np.array([[1/n_classes for _ in range(n_classes)],]).T
    initial_entropy = np.log(n_classes)

    uncalibrated_classifiers = {}
    isotonic_calibrated_classifiers = {}
    sigmoid_calibrated_classifiers = {}

    for n_sensors in range(1, max_sensors+1):
        dataset, feature_names = simdataset.get_dataset(sensor_correlation = sensor_correlation, feature_correlation = feature_correlation,n_sensors = n_sensors, n_total_samples=train_dataset_samples)

        uncalibrated_classifier = classifiers.ClassifierComposition(feature_names, discriminant_model='Gaussian')
        isotonic_calibrated_classifier = classifiers.ClassifierComposition(feature_names, discriminant_model='isotonic_calibration')
        sigmoid_calibrated_classifier = classifiers.ClassifierComposition(feature_names, discriminant_model='sigmoid_calibration')

        uncalibrated_classifier.fit(dataset)
        isotonic_calibrated_classifier.fit(dataset)
        sigmoid_calibrated_classifier.fit(dataset)

        isotonic_calibrated_accuracy = isotonic_calibrated_classifier.expected_confidence(uninformative_prior, n=samples_information)
        sigmoid_calibrated_accuracy = sigmoid_calibrated_classifier.expected_confidence(uninformative_prior, n=samples_information)
        uncalibrated_accuracy = uncalibrated_classifier.expected_confidence(uninformative_prior, n=samples_information)

        # Classifiers are stored as (classifier, information) tuples
        uncalibrated_classifiers[n_sensors] = (uncalibrated_classifier, uncalibrated_accuracy)
        sigmoid_calibrated_classifiers[n_sensors] = (sigmoid_calibrated_classifier, sigmoid_calibrated_accuracy)
        isotonic_calibrated_classifiers[n_sensors] = (isotonic_calibrated_classifier, isotonic_calibrated_accuracy)

    return uncalibrated_classifiers, sigmoid_calibrated_classifiers, isotonic_calibrated_classifiers


def evaluate_classifier(classifiers, sensor_correlation, goal_accuracy, dataset = None):
    initial_entropy = np.log(n_classes)

    # Decide how many sensors should be used
    expected_accuracy = [classifiers[i][1] for i in range(1, max_sensors+1)]

    # expected_posterior_entropy = initial_entropy - information
    selected_number_sensors = [i for i in range(1,max_sensors+1) if expected_accuracy[i-1]>goal_accuracy]
    if selected_number_sensors:
        selected_number_sensors = selected_number_sensors[0]
    else:
        selected_number_sensors = max_sensors
    
    classifier = classifiers[selected_number_sensors][0]
    # Evaluate on test dataset
    
    if dataset is None:
        dataset, _ = simdataset.get_dataset(sensor_correlation = sensor_correlation, n_sensors = selected_number_sensors, n_total_samples=test_dataset_samples)
    
    posteriors = classifier.predict_proba(dataset)
    entropy = stats.entropy(posteriors, base=e, axis=1)

    # Evaluate accuracy on test set
    predictions = posteriors.argmax(axis=1)
    lables = dataset['lable'].to_numpy()
    accuracy = accuracy_score(predictions, dataset['lable'].to_numpy())

    return entropy, selected_number_sensors, accuracy, posteriors, lables, expected_accuracy[selected_number_sensors-1]


def evaluate_goal_accuracy(file_name):

    sensor_correlation = 0.2
    uncalibrated_classifiers, sigmoid_calibrated_classifiers, isotonic_calibrated_classifiers = get_classifiers(sensor_correlation = sensor_correlation, feature_correlation = None)
    dataset, _ = simdataset.get_dataset(sensor_correlation = sensor_correlation, n_sensors = selected_number_sensors, n_total_samples=test_dataset_samples)

    rows = []
    for goal_accuracy in np.linspace(0.5, 1, num=100):
        print(f'Goal accuracy: {goal_accuracy}, Goal entropy: {goal_entropy}')

        uncalibrated_mean_entropy, uncalibrated_selected_number_sensors, uncalibrated_accuracy, uncalibrated_posterior, _, uncalibrated_expected_accuracy = evaluate_classifier(uncalibrated_classifiers, sensor_correlation, goal_accuracy, dataset)
        sigmoid_mean_entropy, sigmoid_selected_number_sensors, sigmoid_accuracy, sigmoid_posterior, _,  sigmoid_expected_accuracy = evaluate_classifier(sigmoid_calibrated_classifiers, sensor_correlation, goal_accuracy, dataset)
        isotonic_mean_entropy, isotonic_selected_number_sensors, isotonic_accuracy, isotonic_posterior, _, isotonic_expected_accuracy = evaluate_classifier(isotonic_calibrated_classifiers, sensor_correlation, goal_accuracy, dataset)
        
        rows.append({'Quantity': 'Realized accuracy', 'Calibration': 'Isotonic regression', 'Selected number of sensors':isotonic_selected_number_sensors,'Accuracy':isotonic_accuracy, 'Mean posterior entropy': isotonic_mean_entropy, 'Sensor correlation': sensor_correlation , 'Goal accuracy': goal_accuracy})
        rows.append({'Quantity': 'Realized accuracy', 'Calibration': 'Logistic regression', 'Selected number of sensors':sigmoid_selected_number_sensors,'Accuracy':sigmoid_accuracy, 'Mean posterior entropy': sigmoid_mean_entropy, 'Sensor correlation': sensor_correlation , 'Goal accuracy': goal_accuracy})
        rows.append({'Quantity': 'Realized accuracy', 'Calibration': 'Uncalibrated', 'Selected number of sensors':uncalibrated_selected_number_sensors,'Accuracy':uncalibrated_accuracy, 'Mean posterior entropy': uncalibrated_mean_entropy, 'Sensor correlation': sensor_correlation , 'Goal accuracy': goal_accuracy})

        rows.append({'Quantity': 'Expected accuracy', 'Calibration': 'Isotonic regression', 'Selected number of sensors':isotonic_selected_number_sensors,'Accuracy':isotonic_expected_accuracy, 'Mean posterior entropy': isotonic_mean_entropy, 'Sensor correlation': sensor_correlation , 'Goal accuracy': goal_accuracy})
        rows.append({'Quantity': 'Expected accuracy', 'Calibration': 'Logistic regression', 'Selected number of sensors':sigmoid_selected_number_sensors,'Accuracy':sigmoid_expected_accuracy, 'Mean posterior entropy': sigmoid_mean_entropy, 'Sensor correlation': sensor_correlation , 'Goal accuracy': goal_accuracy})
        rows.append({'Quantity': 'Expected accuracy', 'Calibration': 'Uncalibrated', 'Selected number of sensors':uncalibrated_selected_number_sensors,'Accuracy':uncalibrated_expected_accuracy, 'Mean posterior entropy': uncalibrated_mean_entropy, 'Sensor correlation': sensor_correlation , 'Goal accuracy': goal_accuracy})

    df = pd.DataFrame(rows)
    pickle.dump( df, open( 'accuracy_goal_accuracy' + file_name, "wb" ) )

    sns.lineplot(data=df, x = 'Goal accuracy', y='Accuracy', markers=False, hue='Calibration', style="Quantity")
    plt.plot([0, 1], [0, 1], "k:", label="Ideal calibration")
    plt.show()

    print('s')



def evaluate_classifiers(file_name):
    
    feature_correlation = None #0.1
    rows = []
    for sensor_correlation in np.linspace(0, 0.6, num=50):
        print(f'Sensor correlation: {sensor_correlation}')
        uncalibrated_classifiers, sigmoid_calibrated_classifiers, isotonic_calibrated_classifiers = get_classifiers(sensor_correlation = sensor_correlation, feature_correlation = feature_correlation)

        uncalibrated_mean_entropy, uncalibrated_selected_number_sensors, uncalibrated_accuracy, uncalibrated_posterior, _, uncalibrated_expected_accuracy = evaluate_classifier(uncalibrated_classifiers, sensor_correlation, goal_accuracy)
        sigmoid_mean_entropy, sigmoid_selected_number_sensors, sigmoid_accuracy, sigmoid_posterior, _,  sigmoid_expected_accuracy = evaluate_classifier(sigmoid_calibrated_classifiers, sensor_correlation, goal_accuracy)
        isotonic_mean_entropy, isotonic_selected_number_sensors, isotonic_accuracy, isotonic_posterior, _, isotonic_expected_accuracy = evaluate_classifier(isotonic_calibrated_classifiers, sensor_correlation, goal_accuracy)

        rows.append({'Quantity': 'Realized accuracy', 'Calibration': 'Isotonic regression', 'Selected number of sensors':isotonic_selected_number_sensors,'Accuracy':isotonic_accuracy, 'Mean posterior entropy': isotonic_mean_entropy, 'Sensor correlation': sensor_correlation })
        rows.append({'Quantity': 'Realized accuracy', 'Calibration': 'Logistic regression', 'Selected number of sensors':sigmoid_selected_number_sensors,'Accuracy':sigmoid_accuracy, 'Mean posterior entropy': sigmoid_mean_entropy, 'Sensor correlation': sensor_correlation })
        rows.append({'Quantity': 'Realized accuracy', 'Calibration': 'Uncalibrated', 'Selected number of sensors':uncalibrated_selected_number_sensors,'Accuracy':uncalibrated_accuracy, 'Mean posterior entropy': uncalibrated_mean_entropy, 'Sensor correlation': sensor_correlation })

        rows.append({'Quantity': 'Expected accuracy', 'Calibration': 'Isotonic regression', 'Selected number of sensors':isotonic_selected_number_sensors,'Accuracy':isotonic_expected_accuracy, 'Mean posterior entropy': isotonic_mean_entropy, 'Sensor correlation': sensor_correlation })
        rows.append({'Quantity': 'Expected accuracy', 'Calibration': 'Logistic regression', 'Selected number of sensors':sigmoid_selected_number_sensors,'Accuracy':sigmoid_expected_accuracy, 'Mean posterior entropy': sigmoid_mean_entropy, 'Sensor correlation': sensor_correlation })
        rows.append({'Quantity': 'Expected accuracy', 'Calibration': 'Uncalibrated', 'Selected number of sensors':uncalibrated_selected_number_sensors,'Accuracy':uncalibrated_expected_accuracy, 'Mean posterior entropy': uncalibrated_mean_entropy, 'Sensor correlation': sensor_correlation })

        rows.append({'Quantity': 'Predicted accuracy', 'Calibration': 'Isotonic regression', 'Selected number of sensors':isotonic_selected_number_sensors,'Accuracy':np.mean(isotonic_posterior), 'Mean posterior entropy': isotonic_mean_entropy, 'Sensor correlation': sensor_correlation })
        rows.append({'Quantity': 'Predicted accuracy', 'Calibration': 'Logistic regression', 'Selected number of sensors':sigmoid_selected_number_sensors,'Accuracy':np.mean(sigmoid_posterior), 'Mean posterior entropy': sigmoid_mean_entropy, 'Sensor correlation': sensor_correlation })
        rows.append({'Quantity': 'Predicted accuracy', 'Calibration': 'Uncalibrated', 'Selected number of sensors':uncalibrated_selected_number_sensors,'Accuracy':np.mean(uncalibrated_posterior), 'Mean posterior entropy': uncalibrated_mean_entropy, 'Sensor correlation': sensor_correlation })

        print("s")
        
    df = pd.DataFrame(rows)

    print('s')
    pickle.dump( df, open( file_name, "wb" ) )

    sns.lineplot(data=df, x = 'Sensor correlation', y='Accuracy', markers=False, hue='Calibration', style="Quantity")
    plt.show()

    sns.lineplot(data=df, x = 'Sensor correlation', y='Mean posterior entropy', markers=False, dashes=False, hue='Calibration', style="Calibration")
    plt.show()

    sns.lineplot(data=df, x = 'Sensor correlation', y='Selected number of sensors', markers=False, dashes=False, hue='Calibration', style="Calibration")
    plt.show()


if __name__=='__main__':
    file_name = 'Uncorrelated.df'
    evaluate_goal_accuracy(file_name)
    # render_reliability()
    # evaluate_n_sensors()
    # Anteckningar från meeting with nicolas. 
    # Nått är fel med riktig accuracy och entropy measurements.
    # öka diskretizeringen i entropy to accuracy.
    # Kanske inte ta mean av entropy, ta mean av confidence....
    # Skapa samma graf över goal accuracy för simulated results. 
    # Ta bort predicted accuracy