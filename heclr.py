import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('134008.jpg')

img = cv2.resize(img, (256, 256))
cv2.imshow('Input Image', img)

b, g, r = cv2.split(img)

freq_b = np.zeros(256)
freq_g = np.zeros(256)
freq_r = np.zeros(256)

for i in range(256):
    for j in range(256):
        freq_b[b[i, j]] += 1
        freq_g[g[i, j]] += 1
        freq_r[r[i, j]] += 1

plt.plot(np.arange(256), freq_b, color='blue')
plt.grid(True)
plt.show()

plt.plot(np.arange(256), freq_g, color='green')
plt.grid(True)
plt.show()

plt.plot(np.arange(256), freq_r, color='red')
plt.grid(True)
plt.show()

pdf_b = freq_b / (256 * 256)
pdf_g = freq_g / (256 * 256)
pdf_r = freq_r / (256 * 256)

cdf_b = np.zeros(256)
cdf_g = np.zeros(256)
cdf_r = np.zeros(256)

cdf_b[0] = pdf_b[0]
cdf_g[0] = pdf_g[0]
cdf_r[0] = pdf_r[0]

for i in range(1, 256):
    cdf_b[i] = cdf_b[i - 1] + pdf_b[i]
    cdf_g[i] = cdf_g[i - 1] + pdf_g[i]
    cdf_r[i] = cdf_r[i - 1] + pdf_r[i]

cdf_b = cdf_b * 255
cdf_g = cdf_g * 255
cdf_r = cdf_r * 255

better_b = np.zeros_like(b)
better_g = np.zeros_like(g)
better_r = np.zeros_like(r)

for i in range(256):
    for j in range(256):
        better_b[i, j] = cdf_b[b[i, j]]
        better_g[i, j] = cdf_g[g[i, j]]
        better_r[i, j] = cdf_r[r[i, j]]

better_img = cv2.merge((better_b, better_g, better_r))

freqtemp_b = np.zeros(256)
freqtemp_g = np.zeros(256)
freqtemp_r = np.zeros(256)

for i in range(256):
    for j in range(256):
        freqtemp_b[better_b[i, j]] += 1
        freqtemp_g[better_g[i, j]] += 1
        freqtemp_r[better_r[i, j]] += 1

plt.plot(np.arange(256), freqtemp_b, color='blue')
plt.grid(True)
plt.show()

plt.plot(np.arange(256), freqtemp_g, color='green')
plt.grid(True)
plt.show()

plt.plot(np.arange(256), freqtemp_r, color='red')
plt.grid(True)
plt.show()

cv2.imwrite('134008he.jpg', better_img)
cv2.imshow('134008he.jpg', better_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
