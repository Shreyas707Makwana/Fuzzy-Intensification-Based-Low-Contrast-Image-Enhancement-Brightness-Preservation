import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import signal
from skimage.metrics import structural_similarity as ssim
from skimage.util import img_as_float

def calculate_ssim(img1, img2):
    return ssim(img1, img2, data_range=img2.max() - img2.min())

def calculate_vsi(img1, img2):
    def simple_saliency_map(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(gx**2 + gy**2)
        return mag / np.max(mag)

    smap1 = simple_saliency_map(img1)
    smap2 = simple_saliency_map(img2)
    
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    smap = (smap1 + smap2) / 2
    
    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return np.mean(ssim_map * smap)

def calculate_mmssim(img1, img2, max_scale=5):
    weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    mssim = []
    mcs = []
    for scale in range(max_scale):
        ssim_map, cs_map = ssim(img1, img2, win_size=11, data_range=255, full=True)
        mssim.append(np.mean(ssim_map))
        mcs.append(np.mean(cs_map))
        
        filtered_im1 = cv2.GaussianBlur(img1, (5, 5), 1.5)
        filtered_im2 = cv2.GaussianBlur(img2, (5, 5), 1.5)
        img1 = cv2.resize(filtered_im1, (filtered_im1.shape[1] // 2, filtered_im1.shape[0] // 2))
        img2 = cv2.resize(filtered_im2, (filtered_im2.shape[1] // 2, filtered_im2.shape[0] // 2))
    
    return np.prod(np.array(mcs[:-1]) ** weights[:-1]) * (mssim[-1] ** weights[-1])

def calculate_gmsd(img1, img2):

    img1 = img_as_float(img1)
    img2 = img_as_float(img2)
    sobelx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8
    sobely = sobelx.T
    grad1_x = signal.convolve2d(img1, sobelx, mode='same', boundary='symm')
    grad1_y = signal.convolve2d(img1, sobely, mode='same', boundary='symm')
    grad2_x = signal.convolve2d(img2, sobelx, mode='same', boundary='symm')
    grad2_y = signal.convolve2d(img2, sobely, mode='same', boundary='symm')
    
   
    grad_mag1 = np.sqrt(grad1_x**2 + grad1_y**2)
    grad_mag2 = np.sqrt(grad2_x**2 + grad2_y**2)
    
  
    c = 0.0026  
    gms = (2 * grad_mag1 * grad_mag2 + c) / (grad_mag1**2 + grad_mag2**2 + c)
    gmsd = np.sqrt(np.mean((gms - gms.mean())**2))
    
    return gmsd



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

img = cv2.imread('7.1.03.tiff', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (256, 256))
cv2.imshow('Input Image', img)

mean_val = np.mean(img)

low_part = img[img <= mean_val]
high_part = img[img > mean_val]

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
print(low_part.size)
print(high_part.size)
print(low_part.size+high_part.size)

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

freqtemp = np.zeros(256)

for i in range(256):
    for j in range(256):
        freqtemp[better_img[i, j]] += 1

plt.plot(np.arange(256), freqtemp, color='green')
plt.grid(True)
plt.show()

cv2.imwrite('better_image_bhe.jpg', better_img)
cv2.imshow('Better Image', better_img)

entropy = calculate_entropy(better_img)
print(f'Entropy of the image: {entropy:.4f}')

psnr = calculate_psnr(img, better_img)
print(f'PSNR of the image: {psnr:.4f} dB')

ambe = calculate_ambe(img, better_img)
print(f'AMBE of the image: {ambe:.4f}')

ssim_value = calculate_ssim(img, better_img)
print(f'SSIM: {ssim_value:.4f}')

vsi_value = calculate_vsi(img, better_img)
print(f'VSI: {vsi_value:.4f}')

mmssim_value = calculate_mmssim(img, better_img)
print(f'MMSSIM: {mmssim_value:.4f}')

gmsd_value = calculate_gmsd(img, better_img)
print(f'GMSD: {gmsd_value:.4f}')

cv2.waitKey(0)
cv2.destroyAllWindows()
