from math import e

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from scipy import stats

def confidence_ellipse(mean, cov, ax, n_std=3.0, facecolor="none", **kwargs):
    """
    """
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
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



class InverseEntropy:
    def __init__(self):
        
        confidence = np.linspace(0.5, 1, num=10000)
        entropy = stats.entropy(np.stack((confidence, 1-confidence)), base=e, axis=0)
        self.inverse = np.stack((entropy, confidence)).T
    
    def inverse_entropy(self, entropy):
        if isinstance(entropy, np.ndarray):
            entropy = entropy.squeeze()
            confidence = []
            for value in entropy:
                index = np.abs(self.inverse[:,0] - value).argmin()
                confidence.append(self.inverse[index,1])
            confidence = np.array(confidence)
        else:
            index = np.abs(self.inverse[:,0] - entropy).argmin()
            confidence = self.inverse[index,1]
        return confidence 
        
        