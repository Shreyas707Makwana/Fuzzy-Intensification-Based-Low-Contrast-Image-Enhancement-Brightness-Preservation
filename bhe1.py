import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('Daivik.jpeg', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (256, 256))
cv2.imshow('Input Image', img)

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

median_val = np.median(img)

low_part = img[img <= median_val]
high_part = img[img > median_val]

freq_low = np.zeros(256)
freq_high = np.zeros(256)

for i in range(low_part.shape[0]):
    freq_low[low_part[i]] += 1

for i in range(high_part.shape[0]):
    freq_high[high_part[i]] += 1

plt.plot(np.arange(256), freq_low, color='blue')
plt.grid(True)
plt.show()

plt.plot(np.arange(256), freq_high, color='red')
plt.grid(True)
plt.show()

pdf_low = freq_low / low_part.size
pdf_high = freq_high / high_part.size

cdf_low = np.zeros(256)
cdf_high = np.zeros(256)

cdf_low[0] = pdf_low[0]
cdf_high[0] = pdf_high[0]

for i in range(1, 256):
    cdf_low[i] = cdf_low[i - 1] + pdf_low[i]
    cdf_high[i] = cdf_high[i - 1] + pdf_high[i]

cdf_low = cdf_low * median_val
cdf_high = (cdf_high * (255 - median_val)) + median_val

better_img = np.zeros_like(img)

for i in range(256):
    for j in range(256):
        if img[i, j] <= median_val:
            better_img[i, j] = cdf_low[img[i, j]]
        else:
            better_img[i, j] = cdf_high[img[i, j]]

freqtemp = np.zeros(256)

for i in range(256):
    for j in range(256):
        freqtemp[better_img[i, j]] += 1

plt.plot(np.arange(256), freqtemp, color='green')
plt.grid(True)
plt.show()

cv2.imwrite('better_image_bhe_median.jpg', better_img)
cv2.imshow('Better Image', better_img)

entropy = calculate_entropy(better_img)
print(f'Entropy of the image: {entropy:.4f}')

psnr = calculate_psnr(img, better_img)
print(f'PSNR of the image: {psnr:.4f} dB')

ambe = calculate_ambe(img, better_img)
print(f'AMBE of the image: {ambe:.4f}')


cv2.waitKey(0)
cv2.destroyAllWindows()
