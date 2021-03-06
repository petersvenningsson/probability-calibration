from scipy import stats
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
from numpy import matlib
import pandas as pd

from simconfig import mean_A, mean_B

n_features = 2
n_classes = 2

def set_cross_covariance(covariance_matrix, correlation):
    """ Assumes two-dimensional distribution
    """
    # Avoid edge case if covariance matrix has dtype int
    covariance_matrix = covariance_matrix.astype('float64')
    # set upper triangular
    covariance_matrix[1,0] = correlation*np.sqrt(covariance_matrix[0,0]*covariance_matrix[1,1])

    # Flip covariance matrix
    covariance_matrix = np.tril(covariance_matrix) + np.triu(covariance_matrix.T, 1)
    return covariance_matrix


def get_distributions():
    # Class distribution
    class_imbalance = [0.5, 0.5]
    
    # set_cross_covariance is able to convert to covariance matrices but currently on correlation matrices are used (e.i. normalized to unit variance)
    cov_A = set_cross_covariance(np.diag((1, 1)),0)
    cov_B = set_cross_covariance(np.diag((1,1)), 0)

    class_A = stats.multivariate_normal(mean_A, cov_A)
    class_B = stats.multivariate_normal(mean_B, cov_B)

    distributions = [class_A, class_B]
    return distributions, class_imbalance


def sensor_network_distribution(mean, covariance_matrix, sensor_correlation, n_sensors):

    ## Generate the covariance matrix of the full sensing network
    network_covariance_matrix = block_diag(*(covariance_matrix for _ in range(n_sensors)))
    
    # Covariance between sensors
    sensor_covariance = np.diag((sensor_correlation*covariance_matrix[0,0],sensor_correlation*covariance_matrix[1,1]))
    _sensor_covariance = matlib.repmat(sensor_covariance, n_sensors, n_sensors)

    mask = network_covariance_matrix == 0
    network_covariance_matrix[mask] = _sensor_covariance[mask]

    ## Generate the mean vector
    network_mean = matlib.repmat(mean, 1, n_sensors).squeeze()

    return network_mean, network_covariance_matrix

def get_dataset(sensor_correlation = 0.4, n_sensors = 3, n_total_samples=10000, feature_correlation = None):
    """ Warning, featue_correlation is not meaningful. However, correaltion is 1 for feature correlation = 1 and 
    correlation is 0 for feature_correlation = 0. Check correlation numerically to get true feature_correlation value
    """

    distributions, class_imbalance = get_distributions()

    dataset = None
    for i_class in range(n_classes):
        n_samples = int(n_total_samples*class_imbalance[i_class])
        network_mean, network_cov = sensor_network_distribution(distributions[i_class].mean, distributions[i_class].cov, sensor_correlation, n_sensors)
        network_distribution = stats.multivariate_normal(network_mean, network_cov)

        samples = network_distribution.rvs(size = n_samples)

        if feature_correlation:
            seed_feature = samples[:,:]
            for i_sensor in range(n_sensors): # Bit messy here, apologies.

                feature_index_1 = i_sensor*2
                feature_index_2 = i_sensor*2 + 1
                # Correlated features are generated by mixing independent normally distributed variables. 
                # sqrt is used taken to give samples unit variance.
                correlated_feature_1 = seed_feature[:,feature_index_1]*np.sqrt(feature_correlation/2) + seed_feature[:,feature_index_2]*np.sqrt((1 - feature_correlation/2))
                correlated_feature_2 = seed_feature[:,feature_index_2]*np.sqrt(feature_correlation/2) + seed_feature[:,feature_index_1]*np.sqrt((1 - feature_correlation/2))

                samples[:,feature_index_1] = correlated_feature_1
                samples[:,feature_index_2] = correlated_feature_2

        lables = np.full((n_samples, 1), i_class)
        samples = np.hstack((samples, lables))

        if dataset is None:
            dataset = samples
        else:
            dataset = np.vstack((dataset, samples))

    feature_names = [f'feature_{i}' for i in range(n_sensors*n_features)]
    columns = [*feature_names, 'lable']

    return pd.DataFrame(dataset, columns=columns), feature_names
    


def confidence_ellipse(mean, cov, ax, n_std=3.0, facecolor="none", **kwargs):
    """
    """
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])

    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse(
        (0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs
    )

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = mean[0]

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = mean[1]

    transf = (
        transforms.Affine2D()
        .rotate_deg(45)
        .scale(scale_x, scale_y)
        .translate(mean_x, mean_y)
    )

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def render_distribution():

    distributions, _ = get_distributions()
    distributions_colors = ['tab:green', 'tab:orange', 'tab:red']

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.axes.set_title("Feature distributions")
    for c, color in zip(distributions, distributions_colors):
        confidence_ellipse(
            np.array(c.mean),
            np.array(c.cov),
            ax,
            n_std=2.0,
            facecolor="none",
            edgecolor=color,
            linewidth=2,
        )
    plt.xlim((-5,5))
    plt.ylim((-2.5,5))
    plt.legend(['Class A', 'Class B', 'Class C'])
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()


if __name__=='__main__':
    render_distribution()
    render_distribution()