import numpy as np


def mutual_information(hgram):
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1)  # marginal for x over y
    py = np.sum(pxy, axis=0)  # marginal for y over x
    px_py = px[:, None] * py[None, :]  # Broadcast to multiply marginals
    nzs = pxy > 0  # Only non-zero pxy values contribute to the sum
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))


def MutualInfoImage(image, gt):
    hist_2d, x_edges, y_edges = np.histogram2d(image.ravel(), gt.ravel(), bins=20)
    mutual = mutual_information(hist_2d)
    return mutual




