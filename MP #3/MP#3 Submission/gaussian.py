import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import binary_closing, binary_opening
from scipy.stats import multivariate_normal
import os

def CCL(mat):
    d = (mat > 0).astype(int)
    h, w = d.shape
    res = np.zeros((h, w), dtype=int)
    nid = 1
    eqs = []

    for y in range(h):
        for x in range(w):
            if d[y, x] == 1:
                nbs = set()
                if x > 0 and res[y, x-1] > 0: nbs.add(res[y, x-1])
                if y > 0 and res[y-1, x] > 0: nbs.add(res[y-1, x])
                if y > 0 and x > 0 and res[y-1, x-1] > 0: nbs.add(res[y-1, x-1])
                if y > 0 and x < w-1 and res[y-1, x+1] > 0: nbs.add(res[y-1, x+1])

                if not nbs:
                    res[y, x] = nid
                    eqs.append({nid})
                    nid += 1
                else:
                    cl = min(nbs)
                    res[y, x] = cl
                    if len(nbs) > 1:
                        ll = list(nbs)
                        for i in range(1, len(ll)):
                            sa = next(s for s in eqs if ll[0] in s)
                            sb = next(s for s in eqs if ll[i] in s)
                            if sa is not sb:
                                sa.update(sb)
                                eqs.remove(sb)

    eqs.sort(key=lambda s: min(s))
    m = {}
    for fid, s in enumerate(eqs, 1):
        for oid in s: m[oid] = fid

    for y in range(h):
        for x in range(w):
            if res[y, x] > 0: res[y, x] = m[res[y, x]]

    return res, len(eqs)

def get_hs(img):
    arr = np.array(Image.fromarray(img).convert('HSV')).astype(np.float64)
    return arr[:,:,0] / 255.0, arr[:,:,1] / 255.0

def mahal_dist(pts, mu, cov_inv):
    d = pts - mu
    return np.sum(d @ cov_inv * d, axis=1)

def run_segment(img, mu, cov_inv, thresh=2.25):
    hs, sv = get_hs(img)
    pts = np.column_stack((hs.ravel(), sv.ravel()))
    
    md = mahal_dist(pts, mu, cov_inv).reshape(hs.shape)
    raw = (md < thresh).astype(np.uint8)

    raw = binary_closing(raw, structure=np.ones((9, 9))).astype(np.uint8)
    raw = binary_opening(raw, structure=np.ones((3, 3))).astype(np.uint8)

    labeled, nc = CCL(raw)
    if nc == 0:
        return np.zeros_like(raw), np.zeros((*raw.shape, 3), dtype=np.uint8)

    sizes = [np.sum(labeled == i) for i in range(1, nc + 1)]
    best = (labeled == np.argmax(sizes) + 1).astype(np.uint8)

    overlay = img.copy()
    overlay[best == 0] = 0
    return best * 255, overlay

def main():
    tr, te, od = 'training', 'testing', 'results_mp3_gaussian'
    if not os.path.exists(od): os.makedirs(od)

    samples = []
    if os.path.exists(tr):
        for fn in [f for f in os.listdir(tr)]:
            img = np.array(Image.open(os.path.join(tr, fn)).convert('RGB'))
            r, g, b = img[:,:,0].astype(int), img[:,:,1].astype(int), img[:,:,2].astype(int)
            mask = (r > 95) & (g > 40) & (b > 20) & \
                   (r > g) & (r > b) & \
                   (np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b) > 15) & \
                   (np.abs(r - g) > 15)
            if np.any(mask):
                hs, sv = get_hs(img)
                samples.append(np.column_stack((hs[mask], sv[mask])))

    if not samples: return

    all_px = np.vstack(samples)
    mu = np.mean(all_px, axis=0)
    cov = np.cov(all_px, rowvar=False)
    cov_inv = np.linalg.inv(cov)

    if os.path.exists(te):
        for fn in [f for f in os.listdir(te) if f.lower().endswith('.bmp')]:
            arr = np.array(Image.open(os.path.join(te, fn)).convert('RGB'))
            _, overlay = run_segment(arr, mu, cov_inv)

            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            ax[0].imshow(arr); ax[0].set_title(f"Original: {fn}"); ax[0].axis('off')
            ax[1].imshow(overlay); ax[1].set_title("Gaussian Segmentation"); ax[1].axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(od, f"{fn}.jpg"))
            plt.show()
            plt.close()

if __name__ == "__main__":
    main()