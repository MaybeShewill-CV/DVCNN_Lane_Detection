"""
Use RANSAC function fits a line according the candidates extracted by the weights hat-like filter. Use opencv
implemention of RANSAC method
"""
import cv2
try:
    from cv2 import cv2
except ImportError:
    pass
from skimage.measure import LineModelND, ransac


def ransac_linefit_opencv(points):
    """
    Use opencv fitline function to fit the line
    :param points:
    :return: line [vx, vy, x, y] vx, vy represent the direction x, y represent the origin position
    """
    line = cv2.fitLine(points=points, distType=cv2.DIST_WELSCH, param=0, reps=0.01, aeps=0.01)
    return line


def ransac_linefit_sklearn(points):
    """
    Use sklearn ransac function to fit the line
    :param points:
    :return: skimage ransac return the model param set ('origin', 'direction')
    """
    # robustly fit line only using inlier data with RANSAC algorithm
    model_robust, inliers = ransac(points, LineModelND, min_samples=2,
                                   residual_threshold=1, max_trials=1000)
    return model_robust, inliers
