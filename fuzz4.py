import numpy as np
import cv2
import matplotlib.pyplot as plt


img = cv2.imread('5.3.01.tiff', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (256, 256))

cv2.imshow('Input Image', img)


def calculate_fuzzy_dissimilarity_histogram(img):
    rows, cols = img.shape
    std_dev = np.std(img.astype(float))
    hist = np.zeros(256, dtype=float)

    for intensity in range(1, 256): 
        count = 0
        for i in range(2, rows - 1):
            for j in range(2, cols - 1):
                center_val = float(img[i, j])

                if int(center_val) == intensity:
                    neighbors = [
                        float(img[i - 1, j - 1]), float(img[i - 1, j]), float(img[i - 1, j + 1]),
                        float(img[i, j - 1]), center_val, float(img[i, j + 1]),
                        float(img[i + 1, j - 1]), float(img[i + 1, j]), float(img[i + 1, j + 1]),
                    ]
                    similarities = [max(0, 1 - abs(center_val - n) / std_dev) for n in neighbors]
                    fuzzy_dissimilarity_avg = 1 - (sum(similarities) / 9)
                    count += fuzzy_dissimilarity_avg
        hist[intensity] = count

    return hist


def calculate_normal_histogram(img):
    hist = np.zeros(256)
    for i in range(256):
        for j in range(256):
            hist[img[i, j]] += 1
    return hist


fuzzy_hist = calculate_fuzzy_dissimilarity_histogram(img)


normal_hist = calculate_normal_histogram(img)


n = np.count_nonzero(normal_hist)  
adjustment = np.zeros(256)
sumi = np.sum(fuzzy_hist)
print(sumi)

adjustment = (256 * 256 - sumi) / n
print(n)
print(adjustment)


adjusted_hist = normal_hist + adjustment


pdf = adjusted_hist / (256 * 256)


cdf = np.zeros(256)
cdf[0] = pdf[0]
for i in range(1, 256):
    cdf[i] = cdf[i - 1] + pdf[i]



cdf=cdf*255


better_img = np.zeros_like(img)
for i in range(256):
    for j in range(256):
        better_img[i, j] = cdf[img[i, j]]


plt.figure(figsize=(15, 5))


plt.subplot(1, 3, 1)
plt.bar(np.arange(256), normal_hist, color='green', alpha=0.6, label='Normal Histogram')
plt.title('Original Histogram')
plt.xlabel('Intensity Level')
plt.ylabel('Count')
plt.ylim(0,1700)
plt.legend()


plt.subplot(1, 3, 2)
plt.bar(np.arange(256), fuzzy_hist, color='blue', alpha=0.6, label='Fuzzy Histogram')
plt.title('Fuzzy Histogram')
plt.xlabel('Intensity Level')
plt.ylabel('Count')
plt.ylim(0,1700)
plt.legend()

plt.subplot(1, 3, 3)
plt.bar(np.arange(256), adjusted_hist, color='red', alpha=0.7, label='Adjusted Histogram')
plt.title('Adjusted Histogram')
plt.xlabel('Intensity Level')
plt.ylabel('Count')
plt.ylim(0,1700)
plt.legend()


plt.tight_layout()
plt.show()


cv2.imwrite('better_image.jpg', better_img)


cv2.imshow('try', better_img)


cv2.waitKey(0)
cv2.destroyAllWindows()
