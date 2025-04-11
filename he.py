import numpy as np
import cv2

import matplotlib.pyplot as plt

img = cv2.imread('7.1.03.tiff',cv2.IMREAD_GRAYSCALE)

img = cv2.resize(img, (256, 256))
cv2.imshow('Input Image',img)

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


freq=np.zeros(256)

for i in range(256):
    for j in range(256):
        freq[img[i, j]] += 1




plt.plot(np.arange(256), freq,color='blue')
plt.grid(True)
plt.show()


pdf = freq / (256 * 256)


cdf = np.zeros(256)
cdf[0] = pdf[0]
for i in range(1, 256):
    cdf[i] = cdf[i - 1] + pdf[i]



cdf=cdf*255
print(cdf)

better_img = np.zeros_like(img)
for i in range(256):
    for j in range(256):
        better_img[i, j] = cdf[img[i, j]]


freqtemp=np.zeros(256)

for i in range(256):
    for j in range(256):
        freqtemp[better_img[i, j]] += 1


plt.plot(np.arange(256),freqtemp,color='blue')
plt.grid(True)
plt.show()



cv2.imwrite('better_image.jpg', better_img)
cv2.imshow('Better Image',better_img)

entropy = calculate_entropy(img)
print(f'Entropy of the image: {entropy:.4f}')

psnr = calculate_psnr(img, better_img)
print(f'PSNR of the image: {psnr:.4f} dB')

ambe = calculate_ambe(img, better_img)
print(f'AMBE of the image: {ambe:.4f}')

cv2.waitKey(0)
cv2.destroyAllWindows()

