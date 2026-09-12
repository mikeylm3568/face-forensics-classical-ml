import cv2
import numpy as np
from scipy import stats


def extract_baseline_features(img_rgb):
    features = []
    for ch in range(3):
        hist = cv2.calcHist([img_rgb], [ch], None, [32], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        features.extend(hist)
    return np.asarray(features, dtype=np.float32)


def extract_enhanced_features(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    hann = np.outer(np.hanning(h), np.hanning(w))
    windowed = gray.astype(np.float32) * hann
    f_shift = np.fft.fftshift(np.fft.fft2(windowed))
    mag = np.abs(f_shift)
    log_mag = np.log1p(mag)

    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    cutoff = min(h, w) / 4.0

    energy = mag ** 2
    low_e = float(np.sum(energy[dist <= cutoff]))
    high_e = float(np.sum(energy[dist > cutoff]))
    ratio = high_e / (low_e + 1e-8)

    fft_stats = [float(np.mean(log_mag)), float(np.std(log_mag)), ratio]

    blurred = cv2.GaussianBlur(gray, (5, 5), sigmaX=1.0)
    residual = gray.astype(np.float32) - blurred.astype(np.float32)
    res = residual.ravel()

    noise_stats = [
        float(np.mean(res)),
        float(np.std(res)),
        float(stats.skew(res)),
        float(stats.kurtosis(res)),
    ]

    return np.asarray(fft_stats + noise_stats, dtype=np.float32)


def recompress_jpeg(img_rgb, quality):
    params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", bgr, params)
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
