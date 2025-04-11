import numpy as np
import cv2
def calculate_fuzzy_dissimilarity_histogram(img):
    rows, cols = img.shape
    std_dev = np.std(img.astype(float))
    hist = np.zeros(256, dtype=float)
    count=0

    for intensity in range(1, 256):
        for i in range(2, rows - 1):
            for j in range(2, cols - 1):
                center_val = float(img[i, j])
                
                if int(center_val) == intensity:
    
                    a = float(img[i - 1, j - 1])   
                    b = float(img[i - 1, j])       
                    c = float(img[i - 1, j + 1])   
                    d = float(img[i, j - 1])       
                    e = center_val                 
                    f = float(img[i, j + 1])      
                    g = float(img[i + 1, j - 1])   
                    h = float(img[i + 1, j])      
                    i_ = float(img[i + 1, j + 1])  
        
                    a1 = max(0, 1 - abs(e - a) / std_dev)
                    b1 = max(0, 1 - abs(e - b) / std_dev)
                    c1 = max(0, 1 - abs(e - c) / std_dev)
                    d1 = max(0, 1 - abs(e - d) / std_dev)
                    e1 = max(0, 1 - abs(e - e) / std_dev)
                    f1 = max(0, 1 - abs(e - f) / std_dev)
                    g1 = max(0, 1 - abs(e - g) / std_dev)
                    h1 = max(0, 1 - abs(e - h) / std_dev)
                    i1 = max(0, 1 - abs(e - i_) / std_dev)

                    fuzzy_dissimilarity_avg = 1 - ((a1 + b1 + c1 + d1 + e1 + f1 + g1 + h1 + i1) / 9)
                    count += fuzzy_dissimilarity_avg
        hist[intensity]=count
        count=0
    return hist

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

def HE_ALGO(img):
    L = 256 
    w, l = img.shape  
    len_img = w * l
    y = img.flatten()  
    xpdf=calculate_fuzzy_dissimilarity_histogram(img)
    
    Ihist = np.zeros(L)  
    for i in range(L):
        Ihist[i] = xpdf[i]

    Xm = 255  
    HEoutput = np.zeros_like(img)

    C_L = np.zeros(Xm+1) 
    n_L = np.sum(Ihist[0:Xm+1])
    P_L = Ihist[0:Xm+1] / n_L

    C_L[0] = P_L[0]
    for r in range(1, len(P_L)):
        C_L[r] = P_L[r] + C_L[r-1]

    for r in range(w):
        for s in range(l):
            if img[r, s] < (Xm + 1):
                f = Xm * C_L[img[r, s]]
                HEoutput[r, s] = np.round(f)


    if img.dtype == np.uint8:
        HEoutput = HEoutput.astype(np.uint8)
    elif img.dtype == np.uint16:
        HEoutput = HEoutput.astype(np.uint16)
    elif img.dtype == np.int16:
        HEoutput = HEoutput.astype(np.int16)
    elif img.dtype == np.float32:
        HEoutput = HEoutput.astype(np.float32)

    return HEoutput


img = cv2.imread('7.1.03.tiff', cv2.IMREAD_GRAYSCALE) 
result = HE_ALGO(img)
cv2.imshow('Histogram Equalized Image', result)


entropy = calculate_entropy(result)
print(f'Entropy of the image: {entropy:.4f}')

psnr = calculate_psnr(img, result)
print(f'PSNR of the image: {psnr:.4f} dB')

ambe = calculate_ambe(img,result)
print(f'AMBE of the image: {ambe:.4f}')

cv2.waitKey(0)
cv2.destroyAllWindows()

