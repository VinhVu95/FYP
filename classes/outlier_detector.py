import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class OutlierDetector:

    @staticmethod
    def mad_based_outlier(points, thresh=3.5):
        """
        Returns a boolean array with True if points are outliers and False
        otherwise.

        Parameters:
        -----------
            points : An numobservations by numdimensions array of observations
            thresh : The modified z-score to use as a threshold. Observations with
                a modified z-score (based on the median absolute deviation) greater
                than this value will be classified as outliers.

        Returns:
        --------
            mask : A numobservations-length boolean array.

        References:
        ----------
            Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
            Handle Outliers", The ASQC Basic References in Quality Control:
            Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.
        """
        if len(points.shape) == 1:
            points = points[:, None]
        median = np.median(points, axis=0)
        diff = np.sum((points - median) ** 2, axis=-1)
        diff = np.sqrt(diff)
        med_abs_deviation = np.median(diff)

        modified_z_score = 0.6745 * diff / med_abs_deviation

        return modified_z_score > thresh

    @staticmethod
    def percentile_based_outlier(points, threshold=95):
        """

        Args:
            points: data points, array_like object
            threshold: middle of distribution curve

        Returns: points lie at two tails of the curve

        """
        diff = (100 - threshold) / 2.0
        minval, maxval = np.percentile(points, [diff, 100 - diff])
        return (points < minval) | (points > maxval)

    @staticmethod
    def detect_outliers(x):
        fig, axes = plt.subplots(nrows=2)
        for ax, func in zip(axes, [OutlierDetector.percentile_based_outlier, OutlierDetector.mad_based_outlier]):
            sns.distplot(x, ax=ax, rug=True, hist=False)
            outliers = x[func(x)]
            ax.plot(outliers, np.zeros_like(outliers), 'ro', clip_on=False)

        kwargs = dict(y=0.95, x=0.05, ha='left', va='top')
        axes[0].set_title('Percentile-based Outliers', **kwargs)
        axes[1].set_title('Modified z score-based Outliers', **kwargs)
        fig.suptitle('Comparing Outlier Tests with n={}'.format(len(x)), size=14)

        return None


