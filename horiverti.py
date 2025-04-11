import numpy as np
import cv2
import matplotlib.pyplot as plt

def calculate_entropy(image):
    total_pixels = image.size
    freq = np.zeros(256)
    
    for pixel in image.flatten():
        freq[pixel] += 1

    pdf = freq / total_pixels
    epsilon = 1e-10
    entropy = -np.sum(pdf * np.log2(pdf + epsilon))
    
    return entropy

def calculate_psnr(original, processed):
    mse = np.mean((original.astype(np.float32) - processed.astype(np.float32)) ** 2)
    epsilon = 1e-10
    psnr = 10 * np.log10((255 ** 2) / (mse + epsilon))
    
    return psnr

def calculate_ambe(original, processed):
    ambe = np.mean(processed - original)
    return ambe

def bihistogram_equalization_vertical(img):
    mean_val = np.mean(img)

    low_part = img[img <= mean_val]
    high_part = img[img > mean_val]

    freq_low = np.zeros(256)
    freq_high = np.zeros(256)

    for i in range(low_part.shape[0]):
        freq_low[low_part[i]] += 1

    for i in range(high_part.shape[0]):
        freq_high[high_part[i]] += 1

    pdf_low = freq_low / low_part.size
    pdf_high = freq_high / high_part.size

    cdf_low = np.zeros(256)
    cdf_high = np.zeros(256)

    cdf_low[0] = pdf_low[0]
    cdf_high[0] = pdf_high[0]

    for i in range(1, 256):
        cdf_low[i] = cdf_low[i - 1] + pdf_low[i]
        cdf_high[i] = cdf_high[i - 1] + pdf_high[i]

    cdf_low = cdf_low * mean_val
    cdf_high = (cdf_high * (255 - mean_val)) + mean_val

    better_img = np.zeros_like(img)

    for i in range(256):
        for j in range(256):
            if img[i, j] <= mean_val:
                better_img[i, j] = cdf_low[img[i, j]]
            else:
                better_img[i, j] = cdf_high[img[i, j]]

    return better_img, cdf_low, cdf_high

def bihistogram_equalization_horizontal(img):
    freq = np.zeros(256)

    for i in range(256):
        for j in range(256):
            freq[img[i, j]] += 1

    sum_freq = np.sum(freq)
    non_zero_freq = np.count_nonzero(freq)
    avg_freq = sum_freq / non_zero_freq

    freq = np.minimum(freq, avg_freq)

    pdf = freq / (256 * 256)

    cdf = np.zeros(256)
    cdf[0] = pdf[0]
    for i in range(1, 256):
        cdf[i] = cdf[i - 1] + pdf[i]

    cdf = cdf * 255

    better_img = np.zeros_like(img)
    for i in range(256):
        for j in range(256):
            better_img[i, j] = cdf[img[i, j]]

    return better_img, cdf

img = cv2.imread('7.1.03.tiff', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (256, 256))
cv2.imshow('Input Image', img)

vertical_equalized, cdf_low_v, cdf_high_v = bihistogram_equalization_vertical(img)


horizontal_equalized, cdf_h = bihistogram_equalization_horizontal(vertical_equalized)

cv2.imshow('Horizontal Equalized', horizontal_equalized)


entropy = calculate_entropy(horizontal_equalized)
psnr = calculate_psnr(img, horizontal_equalized)
ambe = calculate_ambe(img, horizontal_equalized)

print(f'Entropy of the image: {entropy:.4f}')
print(f'PSNR of the image: {psnr:.4f} dB')
print(f'AMBE of the image: {ambe:.4f}')

plt.figure(figsize=(15, 10))

plt.subplot(231)
plt.hist(img.flatten(), 256, [0, 256], color='blue')
plt.title('Original Image Histogram')

plt.subplot(232)
plt.plot(cdf_low_v, color='green', label='Low CDF (Vertical)')
plt.plot(cdf_high_v, color='orange', label='High CDF (Vertical)')
plt.title('CDF after Vertical Equalization')
plt.legend()

plt.subplot(233)
plt.hist(vertical_equalized.flatten(), 256, [0, 256], color='purple')
plt.title('Vertical Equalized Histogram')

plt.subplot(234)
plt.plot(cdf_h, color='red')
plt.title('CDF after Horizontal Equalization')

plt.subplot(235)
plt.hist(horizontal_equalized.flatten(), 256, [0, 256], color='red')
plt.title('Final Histogram (Horizontal Equalized)')

plt.tight_layout()
plt.show()

cv2.imwrite('bihistogram_equalized.jpg', horizontal_equalized)

cv2.waitKey(0)
cv2.destroyAllWindows()
