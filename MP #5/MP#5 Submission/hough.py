import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import convolve, maximum_filter, label

def gauss_smooth(I, N=11, sigma=2.0):
    half = N // 2
    ax = np.arange(-half, half + 1)
    X, Y = np.meshgrid(ax, ax)
    k = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    k /= k.sum()
    return convolve(I, k, mode='reflect')

def image_gradient(S):
    Kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=float)
    Ky = np.array([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=float)
    Gx = convolve(S, Kx, mode='reflect')
    Gy = convolve(S, Ky, mode='reflect')
    return np.hypot(Gx, Gy), np.arctan2(Gy, Gx)

def nms(Mag, Theta):
    rows, cols = Mag.shape
    out = np.zeros_like(Mag)
    angle = np.rad2deg(Theta) % 180
    for r in range(1, rows-1):
        for c in range(1, cols-1):
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
    return out

def find_threshold(Mag, pct=0.85):
    counts, edges = np.histogram(Mag.ravel(), bins=256)
    cum = np.cumsum(counts) / Mag.size
    idx = min(np.searchsorted(cum, pct), len(edges)-2)
    Thigh = edges[idx+1]
    return 0.5*Thigh, Thigh

def edge_link(Weak, Strong):
    E = Strong.copy().astype(bool)
    visited = Strong.copy().astype(bool)
    stack = list(zip(*np.where(Strong)))
    rows, cols = Strong.shape
    while stack:
        r, c = stack.pop()
        for dr in (-1,0,1):
            for dc in (-1,0,1):
                if dr==0 and dc==0: continue
                nr, nc = r+dr, c+dc
                if 0<=nr<rows and 0<=nc<cols:
                    if Weak[nr,nc] and not visited[nr,nc]:
                        visited[nr,nc] = True
                        E[nr,nc] = True
                        stack.append((nr,nc))
    return E

def canny(I, N=11, sigma=2.0, pct=0.85):
    S = gauss_smooth(I, N, sigma)
    Mag, Theta = image_gradient(S)
    Tlow, Thigh = find_threshold(Mag, pct)
    MS = nms(Mag, Theta)
    return edge_link(MS >= Tlow, MS >= Thigh)

def hough_accumulate(edge_img, K_theta, K_rho=None):
    rows, cols = edge_img.shape
    D = np.sqrt(rows**2 + cols**2)

    if K_rho is None:
        K_rho = K_theta

    thetas = np.linspace(-np.pi/2, np.pi/2, K_theta, endpoint=False)
    rhos   = np.linspace(-D, D, K_rho)
    drho   = rhos[1] - rhos[0]

    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)

    A = np.zeros((K_rho, K_theta), dtype=np.int32)

    ys, xs = np.where(edge_img)
    for x, y in zip(xs, ys):
        rho_vals = x * cos_t + y * sin_t        
        rho_idx  = np.round((rho_vals - rhos[0]) / drho).astype(int)
        valid    = (rho_idx >= 0) & (rho_idx < K_rho)
        A[rho_idx[valid], np.where(valid)[0]] += 1

    return A, rhos, thetas


def detect_peaks(A, threshold_ratio=0.5, neighborhood=10):
    threshold = threshold_ratio * A.max()
    local_max = maximum_filter(A, size=neighborhood)
    peak_mask = (A == local_max) & (A >= threshold)
    peak_rho_idx, peak_theta_idx = np.where(peak_mask)
    votes = A[peak_rho_idx, peak_theta_idx]
    order = np.argsort(votes)[::-1]
    pr_out, pt_out = [], []
    for r, t in zip(peak_rho_idx[order], peak_theta_idx[order]):
        too_close = any(
            abs(r - rr) < neighborhood and abs(t - tt) < neighborhood
            for rr, tt in zip(pr_out, pt_out)
        )
        if not too_close:
            pr_out.append(r)
            pt_out.append(t)
    return np.array(pr_out), np.array(pt_out)


def draw_lines(img_gray, rhos, thetas, peak_rho_idx, peak_theta_idx, max_lines=20):
    rows, cols = img_gray.shape
    out = np.stack([img_gray]*3, axis=-1).astype(np.uint8)

    n = min(max_lines, len(peak_rho_idx))
    for i in range(n):
        rho   = rhos[peak_rho_idx[i]]
        theta = thetas[peak_theta_idx[i]]
        ct, st = np.cos(theta), np.sin(theta)

        pts = []
        if abs(st) > 1e-6:
            for x in [0, cols-1]:
                y = (rho - x*ct) / st
                if 0 <= y < rows:
                    pts.append((int(x), int(y)))
        if abs(ct) > 1e-6:
            for y in [0, rows-1]:
                x = (rho - y*st) / ct
                if 0 <= x < cols:
                    pts.append((int(x), int(y)))

        pts = sorted(set(pts))[:2]
        if len(pts) >= 2:
            x0, y0 = pts[0]
            x1, y1 = pts[1]
            dx, dy = abs(x1-x0), abs(y1-y0)
            sx = 1 if x0 < x1 else -1
            sy = 1 if y0 < y1 else -1
            err = dx - dy
            cx, cy = x0, y0
            while True:
                if 0 <= cx < cols and 0 <= cy < rows:
                    out[cy, cx] = [200, 200, 200]
                if cx == x1 and cy == y1:
                    break
                e2 = 2*err
                if e2 > -dy: err -= dy; cx += sx
                if e2 <  dx: err += dx; cy += sy

    return out


def run_hough_pipeline(I_gray, name, K_theta=180, pct=0.85,
                       peak_ratio=0.5, neighborhood=10, max_lines=20):
    E        = canny(I_gray, pct=pct)
    A, rhos, thetas = hough_accumulate(E, K_theta=K_theta)
    pr, pt   = detect_peaks(A, threshold_ratio=peak_ratio,
                             neighborhood=neighborhood)
    img8     = (I_gray * 255).astype(np.uint8)
    detected = draw_lines(img8, rhos, thetas, pr, pt, max_lines=max_lines)

    sig = np.zeros_like(A, dtype=np.uint8)
    if len(pr):
        sig[pr, pt] = 255

    return E, A, sig, detected

img_names  = ['test', 'test2', 'input']
imgs_gray  = {}
for name in img_names:
    raw = Image.open(f'{name}.bmp').convert('L')
    imgs_gray[name] = np.array(raw, dtype=float) / 255.0

tuning = {
    'test'  : (0.85, 0.20, 25, 4),
    'test2' : (0.85, 0.18, 20, 6),
    'input' : (0.80, 0.65, 20, 4),
}

fig1, axes1 = plt.subplots(3, 5, figsize=(18, 11))
fig1.suptitle('Hough Transform Pipeline – All 3 Images  (K_θ=180)', fontsize=12)
col_titles = ['Input', 'Edge Map', 'Parameter Space (ρ-θ)', 'Significant Intersections', 'Detected Lines']

for i, name in enumerate(img_names):
    pct, pr_ratio, nbr, ml = tuning[name]
    I = imgs_gray[name]
    E, A, sig, detected = run_hough_pipeline(
        I, name, K_theta=180, pct=pct,
        peak_ratio=pr_ratio, neighborhood=nbr, max_lines=ml)

    from scipy.ndimage import binary_dilation
    sig_vis = binary_dilation(sig > 0, iterations=4).astype(float)
    stages = [I, E, A / A.max(), sig_vis, detected]
    cmaps  = ['gray', 'gray', 'hot', 'hot', None]
    for j, (stage, cmap) in enumerate(zip(stages, cmaps)):
        ax = axes1[i, j]
        if cmap:
            ax.imshow(stage, cmap=cmap)
        else:
            ax.imshow(stage)
        if i == 0:
            ax.set_title(col_titles[j], fontsize=9)
        if j == 0:
            ax.set_ylabel(name, fontsize=9)
        if j == 2:
            axes1[i, j].set_xlabel('θ (bins)', fontsize=7)
            axes1[i, j].set_ylabel('ρ (bins)', fontsize=7)
            axes1[i, j].tick_params(labelsize=6)
        else:
            axes1[i, j].axis('off')

plt.tight_layout()
plt.savefig('fig1_hough_pipeline.png', dpi=120, bbox_inches='tight')
plt.show()

K_vals = [45, 90, 180, 360]

fig2, axes2 = plt.subplots(3, len(K_vals), figsize=(16, 11))
fig2.suptitle('Effect of θ Quantization Level on Line Detection', fontsize=12)

for i, name in enumerate(img_names):
    pct, pr_ratio, nbr, ml = tuning[name]
    I = imgs_gray[name]
    E = canny(I, pct=pct)
    img8 = (I * 255).astype(np.uint8)

    for j, K in enumerate(K_vals):
        A, rhos, thetas = hough_accumulate(E, K_theta=K)
        nbr_k = max(6, int(nbr * K / 180))
        ml_k  = tuning[name][3]
        pr, pt = detect_peaks(A, threshold_ratio=pr_ratio, neighborhood=nbr_k)
        detected = draw_lines(img8, rhos, thetas, pr, pt, max_lines=ml_k)
        axes2[i, j].imshow(detected)
        if i == 0:
            axes2[i, j].set_title(f'K_θ = {K}', fontsize=9)
        if j == 0:
            axes2[i, j].set_ylabel(name, fontsize=9)
        axes2[i, j].axis('off')

plt.tight_layout()
plt.savefig('fig2_quantization_comparison.png', dpi=120, bbox_inches='tight')
plt.show()

fig3, axes3 = plt.subplots(3, 3, figsize=(13, 11))
fig3.suptitle('Significant Intersection Detection  (K_θ=180)', fontsize=12)
col3 = ['Accumulator A(ρ,θ)', 'Peaks (threshold + local max)', 'Detected Lines']

for i, name in enumerate(img_names):
    pct, pr_ratio, nbr, ml = tuning[name]
    I = imgs_gray[name]
    E = canny(I, pct=pct)
    A, rhos, thetas = hough_accumulate(E, K_theta=180)
    pr, pt = detect_peaks(A, threshold_ratio=pr_ratio, neighborhood=nbr)
    img8 = (I * 255).astype(np.uint8)
    detected = draw_lines(img8, rhos, thetas, pr, pt, max_lines=ml)

    peak_img = np.zeros_like(A, dtype=float)
    peak_img[pr, pt] = A[pr, pt].astype(float)
    from scipy.ndimage import binary_dilation
    peak_vis = binary_dilation(peak_img > 0, iterations=4).astype(float)
    peak_vis *= peak_img.max()

    stages3 = [A / A.max(), peak_vis / (peak_vis.max() + 1e-9), detected]
    cmaps3  = ['hot', 'hot', None]
    for j, (stage, cmap) in enumerate(zip(stages3, cmaps3)):
        ax = axes3[i, j]
        if cmap:
            ax.imshow(stage, cmap=cmap)
        else:
            ax.imshow(stage)
        if i == 0:
            ax.set_title(col3[j], fontsize=9)
        if j == 0:
            ax.set_ylabel(name, fontsize=9)
        ax.axis('off')

plt.tight_layout()
plt.savefig('fig3_peak_detection.png', dpi=120, bbox_inches='tight')
plt.show()