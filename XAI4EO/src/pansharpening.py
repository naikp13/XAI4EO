import numpy as np
import cv2
from scipy.sparse.linalg import svds
from sklearn.linear_model import Lasso

def gram_schmidt_pansharpen(hsi, msi):
    """Perform Gram-Schmidt pansharpening."""
    hsi = np.asarray(hsi)
    msi = np.asarray(msi)
    C_h, H, W = hsi.shape
    C_m, h, w = msi.shape
    msi_upsampled = np.stack([cv2.resize(msi[b], (W, H), interpolation=cv2.INTER_CUBIC) for b in range(C_m)])
    pan = msi_upsampled.mean(axis=0)
    hsi_2d = hsi.reshape(C_h, -1)
    pan_flat = pan.flatten()[None, :]
    X = np.vstack([pan_flat, hsi_2d])
    Q = np.zeros_like(X)
    R = np.zeros((X.shape[0], X.shape[0]))
    for i in range(X.shape[0]):
        v = X[i]
        for j in range(i):
            R[j, i] = np.dot(Q[j], X[i])
            v = v - R[j, i] * Q[j]
        R[i, i] = np.linalg.norm(v) + 1e-8
        Q[i] = v / R[i, i]
    Q[0] = pan_flat
    fused_2d = np.dot(R.T, Q)
    fused_hsi = fused_2d[1:].reshape(C_h, H, W)
    return fused_hsi

def bsr_pansharpen(hsi, msi, weights=None, alpha=0.1, dict_size=256):
    """Perform Bayesian Sparse Representation (BSR) pansharpening."""
    hsi = np.asarray(hsi, dtype=np.float32)
    msi = np.asarray(msi, dtype=np.float32)
    C_h, H, W = hsi.shape
    C_m, h, w = msi.shape
    hsi = (hsi - hsi.min()) / (hsi.max() - hsi.min() + 1e-8)
    msi = (msi - msi.min()) / (msi.max() - msi.min() + 1e-8)
    msi_upsampled = np.stack([cv2.resize(msi[b], (W, H), interpolation=cv2.INTER_LANCZOS4) for b in range(C_m)])
    if weights is None:
        weights = np.ones(C_m) / C_m
    else:
        weights = np.asarray(weights) / np.sum(weights)
    pan = np.sum(msi_upsampled * weights[:, None, None], axis=0)
    pan = cv2.GaussianBlur(pan, (3, 3), sigmaX=0.5)
    hsi_2d = hsi.reshape(C_h, -1).T
    pan_flat = pan.flatten()
    msi_2d = msi_upsampled.reshape(C_m, -1).T
    D, _, _ = svds(msi_2d, k=min(dict_size, min(msi_2d.shape)))
    lasso = Lasso(alpha=alpha, fit_intercept=False)
    sparse_codes = np.zeros((H * W, D.shape[1]))
    for i in range(H * W):
        lasso.fit(D, hsi_2d[i])
        sparse_codes[i] = lasso.coef_
    pan_codes = np.zeros((H * W, D.shape[1]))
    lasso.fit(D, pan_flat)
    pan_codes = lasso.coef_
    sigma_hsi = np.var(hsi_2d, axis=0) + 1e-8
    sigma_pan = np.var(pan_flat) + 1e-8
    fused_codes = (sigma_pan * sparse_codes + sigma_hsi.mean() * pan_codes[:, None]) / (sigma_pan + sigma_hsi.mean())
    fused_2d = np.dot(fused_codes, D.T).T
    fused_hsi = fused_2d.reshape(C_h, H, W)
    fused_hsi = np.clip(fused_hsi, 0, 1)
    return fused_hsi