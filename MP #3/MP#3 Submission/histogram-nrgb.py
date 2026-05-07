import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import binary_closing, binary_opening
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

def get_nrgb(img):
    f = img.astype(np.float64)
    s = np.sum(f, axis=2, keepdims=True)
    s[s == 0] = 1
    n = f / s
    return n[:,:,0], n[:,:,1]

def build_hist(px, bins=64):
    ri = (px[:, 0] * (bins - 1)).astype(int).clip(0, bins-1)
    gi = (px[:, 1] * (bins - 1)).astype(int).clip(0, bins-1)
    h = np.zeros((bins, bins))
    np.add.at(h, (ri, gi), 1)
    return h / (h.sum() + 1e-7)

def run_segment(img, hst, thresh=None):
    nr, ng = get_nrgb(img)
    bc = hst.shape[0]
    ri = np.clip((nr * (bc - 1)).astype(int), 0, bc - 1)
    gi = np.clip((ng * (bc - 1)).astype(int), 0, bc - 1)
    probs = hst[ri, gi]

    t = hst.max() * 0.05 if thresh is None else thresh
    raw = (probs > t).astype(np.uint8)

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
    tr, te, od = 'training', 'testing', 'results_mp3_nrgb'
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
                nr, ng = get_nrgb(img)
                samples.append(np.column_stack((nr[mask], ng[mask])))

    if not samples: return
    model = build_hist(np.vstack(samples))

    if os.path.exists(te):
        for fn in [f for f in os.listdir(te) if f.lower().endswith('.bmp')]:
            arr = np.array(Image.open(os.path.join(te, fn)).convert('RGB'))
            _, overlay = run_segment(arr, model)

            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            ax[0].imshow(arr); ax[0].set_title(f"Original: {fn}"); ax[0].axis('off')
            ax[1].imshow(overlay); ax[1].set_title("nRGB Histogram Segmentation"); ax[1].axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(od, f"nrgb_{fn}.png"))
            plt.show()
            plt.close()

if __name__ == "__main__":
    main()