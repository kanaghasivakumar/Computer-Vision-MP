import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import csv
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
                if x > 0 and res[y, x-1] > 0:
                    nbs.add(res[y, x-1])
                if y > 0 and res[y-1, x] > 0:
                    nbs.add(res[y-1, x])
                
                if not nbs:
                    res[y, x] = nid
                    eqs.append({nid})
                    nid += 1
                else:
                    cl = min(nbs)
                    res[y, x] = cl
                    if len(nbs) > 1:
                        ll = list(nbs)
                        v1, v2 = ll[0], ll[1]
                        s_a = next(s for s in eqs if v1 in s)
                        s_b = next(s for s in eqs if v2 in s)
                        if s_a is not s_b:
                            s_a.update(s_b)
                            eqs.remove(s_b)

    eqs.sort(key=lambda s: min(s))
    m = {}
    for fid, s in enumerate(eqs, 1):
        for oid in s:
            m[oid] = fid
            
    for y in range(h):
        for x in range(w):
            if res[y, x] > 0:
                res[y, x] = m[res[y, x]]
                
    return res, len(eqs)

def main():
    rd = 'results'
    if not os.path.exists(rd):
        os.makedirs(rd)

    fl = [f for f in os.listdir('.') if f.lower().endswith('.bmp')]
    
    for fn in fl:
        with Image.open(fn) as i:
            gray = i.convert('L')
            raw = np.array(gray)
            
        b = (raw > 128).astype(int)
        t, n = CCL(b)

        if "gun" in fn.lower():
            sz = 25
            ref = np.zeros_like(t)
            vc = 0
            for i in range(1, n + 1):
                msk = (t == i)
                if np.sum(msk) >= sz:
                    vc += 1
                    ref[msk] = vc
            out, tot, lbl = ref, vc, "Filtered"
        else:
            out, tot, lbl = t, n, "Raw"

        cp = os.path.join(rd, fn.replace('.bmp', '_labels.csv'))
        with open(cp, 'w', newline='') as f:
            csv.writer(f).writerows(out)

        plt.figure(figsize=(7, 7))
        v = out.astype(float)
        v[v == 0] = np.nan
        
        plt.imshow(v, cmap='gist_rainbow', interpolation='nearest')
        plt.title(f"{fn} | {lbl} count: {tot}")
        plt.axis('off')
        
        ip = os.path.join(rd, fn.replace('.bmp', '_plot.png'))
        plt.savefig(ip)
        plt.show()
        plt.close()

if __name__ == "__main__":
    main()