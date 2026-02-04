import numpy as np
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

def feature_extraction(img: np.ndarray, distances = [1], angles = [0]):
    glcm_features = []
    lbp_features = []

    img_u8 = (img * 255).astype(np.uint8) if img.max() <= 1 else img.astype(np.uint8)

    # GLCM
    glcm = graycomatrix(img_u8,
                        distances=distances,
                        angles=angles,
                        levels=256,
                        symmetric=True,
                        normed=True)

    contrast = graycoprops(glcm, 'contrast')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    glcm_vector = [contrast, energy, homogeneity, correlation]
    glcm_features.append(glcm_vector)

    # LBP
    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(img_u8, n_points, radius, method='uniform')
    hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, n_points + 3),
        range=(0, n_points + 2)
    )
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    lbp_features.append(hist)

    glcm_features = np.asarray(glcm_features)
    lbp_features = np.asarray(lbp_features)
    combined = np.hstack((glcm_features, lbp_features))

    return combined