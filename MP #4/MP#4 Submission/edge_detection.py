import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import convolve, map_coordinates

sys.setrecursionlimit(100000)

def GaussSmoothing(I, N, Sigma):
    half = N // 2
    ax = np.arange(-half, half + 1)
    X, Y = np.meshgrid(ax, ax)
    kernel = np.exp(-(X**2 + Y**2) / (2 * Sigma**2))
    kernel /= kernel.sum()
    return convolve(I, kernel, mode='reflect')

def ImageGradient(S):
    Kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=float)
    Ky = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]], dtype=float)
    Gx    = convolve(S, Kx, mode='reflect')
    Gy    = convolve(S, Ky, mode='reflect')
    Mag   = np.hypot(Gx, Gy)
    Theta = np.arctan2(Gy, Gx)
    return Mag, Theta

def FindThreshold(Mag, percentageOfNonEdge):
    counts, bin_edges = np.histogram(Mag.ravel(), bins=256)
    cum = np.cumsum(counts) / Mag.size
    idx = np.searchsorted(cum, percentageOfNonEdge)
    idx = min(idx, len(bin_edges) - 2)
    Thigh = bin_edges[idx + 1]
    Tlow  = 0.5 * Thigh
    return Tlow, Thigh

def NonmaximaSupress(Mag, Theta, method='quantize'):
    rows, cols = Mag.shape
    out   = np.zeros_like(Mag)
    angle = np.rad2deg(Theta) % 180   

    if method == 'quantize':
        for r in range(1, rows - 1):
            for c in range(1, cols - 1):
                a = angle[r, c]
                m = Mag[r, c]
                if a < 22.5 or a >= 157.5:       
                    n1, n2 = Mag[r, c-1], Mag[r, c+1]
                elif a < 67.5:                    
                    n1, n2 = Mag[r-1, c+1], Mag[r+1, c-1]
                elif a < 112.5:                     
                    n1, n2 = Mag[r-1, c], Mag[r+1, c]
                else:                              
                    n1, n2 = Mag[r-1, c-1], Mag[r+1, c+1]
                if m >= n1 and m >= n2:
                    out[r, c] = m

    elif method == 'interpolate':
        for r in range(1, rows - 1):
            for c in range(1, cols - 1):
                dx = np.cos(Theta[r, c])
                dy = np.sin(Theta[r, c])
                m  = Mag[r, c]
                n1 = map_coordinates(Mag, [[r + dy], [c + dx]],
                                     order=1, mode='reflect')[0]
                n2 = map_coordinates(Mag, [[r - dy], [c - dx]],
                                     order=1, mode='reflect')[0]
                if m >= n1 and m >= n2:
                    out[r, c] = m

    return out

def EdgeLinking(Weak, Strong):
    rows, cols = Strong.shape
    E       = Strong.copy().astype(bool)
    visited = Strong.copy().astype(bool)
    stack   = list(zip(*np.where(Strong)))

    while stack:
        r, c = stack.pop()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if Weak[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        E[nr, nc]       = True
                        stack.append((nr, nc))
    return E

def run_canny(I, N=11, Sigma=2.0, pct=0.85, nms='quantize'):
    S            = GaussSmoothing(I, N, Sigma)
    Mag, Theta   = ImageGradient(S)
    Tlow, Thigh  = FindThreshold(Mag, pct)
    MagSup       = NonmaximaSupress(Mag, Theta, nms)
    E            = EdgeLinking(MagSup >= Tlow, MagSup >= Thigh)
    return E

def sobel_edges(I, pct=0.85):
    Kx  = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=float)
    Ky  = np.array([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=float)
    mag = np.hypot(convolve(I, Kx, mode='reflect'),
                   convolve(I, Ky, mode='reflect'))
    return mag > np.percentile(mag, pct * 100)

def roberts_edges(I, pct=0.85):
    Kr1 = np.array([[1, 0],[0,-1]], dtype=float)
    Kr2 = np.array([[0, 1],[-1,0]], dtype=float)
    mag = np.hypot(convolve(I, Kr1, mode='reflect'),
                   convolve(I, Kr2, mode='reflect'))
    return mag > np.percentile(mag, pct * 100)


def zerocross_edges(I, Sigma=2.0):
    from scipy.ndimage import gaussian_laplace
    log = gaussian_laplace(I, sigma=Sigma)
    zc  = np.zeros_like(log, dtype=bool)
    zc[1:,  :] |= (log[1:, :]  * log[:-1, :] < 0)
    zc[:,  1:] |= (log[:, 1:]  * log[:, :-1] < 0)
    return zc

img_names = ['joy1', 'pointer1', 'test1', 'lena']
imgs = {}
for name in img_names:
    raw = Image.open(f'{name}.bmp').convert('L')
    imgs[name] = np.array(raw, dtype=float) / 255.0

TITLE_FS  = 11
LABEL_FS  = 9
TICK_OFF  = True

param_combos = [(5, 1.0), (5, 4.0), (11, 1.0), (11, 2.0), (21, 2.0), (21, 4.0)]
n_combos     = len(param_combos)
n_imgs       = len(img_names)

fig1, axes1 = plt.subplots(n_imgs, n_combos, figsize=(18, 12))
fig1.suptitle('Canny: varying N and σ  (pct=0.85, NMS=quantize)', fontsize=TITLE_FS)

for i, name in enumerate(img_names):
    for j, (N, Sig) in enumerate(param_combos):
        E = run_canny(imgs[name], N=N, Sigma=Sig, pct=0.85)
        axes1[i, j].imshow(E, cmap='gray')
        if i == 0:
            axes1[i, j].set_title(f'N={N}, σ={Sig}', fontsize=LABEL_FS)
        if j == 0:
            axes1[i, j].set_ylabel(name, fontsize=LABEL_FS)
        axes1[i, j].axis('off')

plt.tight_layout()
plt.savefig('fig1_vary_N_sigma.png', dpi=120, bbox_inches='tight')
plt.show()

pct_vals = [0.80, 0.85, 0.90, 0.95]

fig2, axes2 = plt.subplots(n_imgs, len(pct_vals), figsize=(14, 12))
fig2.suptitle('Canny: varying percentageOfNonEdge  (N=11, σ=2)', fontsize=TITLE_FS)

for i, name in enumerate(img_names):
    for j, pct in enumerate(pct_vals):
        E = run_canny(imgs[name], N=11, Sigma=2.0, pct=pct)
        axes2[i, j].imshow(E, cmap='gray')
        if i == 0:
            axes2[i, j].set_title(f'pct = {pct}', fontsize=LABEL_FS)
        if j == 0:
            axes2[i, j].set_ylabel(name, fontsize=LABEL_FS)
        axes2[i, j].axis('off')

plt.tight_layout()
plt.savefig('fig2_vary_pct.png', dpi=120, bbox_inches='tight')
plt.show()

fig3, axes3 = plt.subplots(n_imgs, 2, figsize=(8, 14))
fig3.suptitle('NMS method comparison  (N=11, σ=2, pct=0.85)', fontsize=TITLE_FS)

for i, name in enumerate(img_names):
    for j, method in enumerate(['quantize', 'interpolate']):
        E = run_canny(imgs[name], N=11, Sigma=2.0, pct=0.85, nms=method)
        axes3[i, j].imshow(E, cmap='gray')
        if i == 0:
            axes3[i, j].set_title(f'NMS: {method}', fontsize=LABEL_FS)
        if j == 0:
            axes3[i, j].set_ylabel(name, fontsize=LABEL_FS)
        axes3[i, j].axis('off')

plt.tight_layout()
plt.savefig('fig3_nms_comparison.png', dpi=120, bbox_inches='tight')
plt.show()

fig4, axes4 = plt.subplots(n_imgs, 4, figsize=(14, 14))
fig4.suptitle('Thresholding & Edge Linking stages  (N=11, σ=2, pct=0.85)', fontsize=TITLE_FS)
col_titles = ['Gradient Magnitude', 'Strong Edges (Thigh)',
              'Weak Edges (Tlow)', 'Final (Linked)']

for i, name in enumerate(img_names):
    I            = imgs[name]
    S            = GaussSmoothing(I, 11, 2.0)
    Mag, Theta   = ImageGradient(S)
    Tlow, Thigh  = FindThreshold(Mag, 0.85)
    MagSup       = NonmaximaSupress(Mag, Theta, 'quantize')
    Strong       = MagSup >= Thigh
    Weak         = MagSup >= Tlow
    Linked       = EdgeLinking(Weak, Strong)

    stages = [Mag / Mag.max(), Strong, Weak, Linked]
    for j, stage in enumerate(stages):
        axes4[i, j].imshow(stage, cmap='gray')
        if i == 0:
            axes4[i, j].set_title(col_titles[j], fontsize=LABEL_FS)
        if j == 0:
            axes4[i, j].set_ylabel(name, fontsize=LABEL_FS)
        axes4[i, j].axis('off')

plt.tight_layout()
plt.savefig('fig4_edge_linking_stages.png', dpi=120, bbox_inches='tight')
plt.show()

detector_labels = ['Canny', 'Sobel', 'Roberts', 'Zero-cross']
detector_fns    = [
    lambda I: run_canny(I, N=11, Sigma=2.0, pct=0.85),
    lambda I: sobel_edges(I, pct=0.85),
    lambda I: roberts_edges(I, pct=0.85),
    lambda I: zerocross_edges(I, Sigma=2.0),
]

fig5, axes5 = plt.subplots(n_imgs, 4, figsize=(14, 14))
fig5.suptitle('Detector comparison – Canny vs Sobel vs Roberts vs Zero-cross', fontsize=TITLE_FS)

for i, name in enumerate(img_names):
    for j, (label, fn) in enumerate(zip(detector_labels, detector_fns)):
        E = fn(imgs[name])
        axes5[i, j].imshow(E, cmap='gray')
        if i == 0:
            axes5[i, j].set_title(label, fontsize=LABEL_FS)
        if j == 0:
            axes5[i, j].set_ylabel(name, fontsize=LABEL_FS)
        axes5[i, j].axis('off')

plt.tight_layout()
plt.savefig('fig5_detector_comparison.png', dpi=120, bbox_inches='tight')
plt.show()