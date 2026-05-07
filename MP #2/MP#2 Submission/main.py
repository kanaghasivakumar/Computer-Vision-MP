import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def hst_eq(arr):
    hst, _ = np.histogram(arr.flatten(), 256, [0, 256])
    pdf = hst / hst.sum()
    cdf = pdf.cumsum()
    lut = np.round(cdf * 255).astype(np.uint8)
    return lut[arr]

def fit_surf(arr, mode='linear'):
    rows, cols = arr.shape
    v_idx, u_idx = np.indices((rows, cols))
    u_f = u_idx.flatten()
    v_f = v_idx.flatten()
    trg = arr.flatten().astype(np.float64)

    if mode == 'linear':
        mtx = np.column_stack((u_f, v_f, np.ones_like(u_f)))
    else:
        mtx = np.column_stack((u_f**2, v_f**2, u_f*v_f, u_f, v_f, np.ones_like(u_f)))

    mtx_t = mtx.T
    sol = np.linalg.solve(mtx_t @ mtx, mtx_t @ trg)
    surf = (mtx @ sol).reshape(rows, cols)
    return surf

def scale_img(arr, surf):
    res = arr - surf
    res -= res.min()
    return (res * (255 / res.max())).astype(np.uint8)

def trunc_img(arr, surf):
    res = arr - surf + np.mean(arr)
    return np.clip(res, 0, 255).astype(np.uint8)

def save_plt(img, title, fn, path):
    plt.figure()
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.title(title)
    plt.axis('off')
    plt.savefig(os.path.join(path, fn))
    plt.close()

def main():
    rd = 'results_mp2'
    if not os.path.exists(rd):
        os.makedirs(rd)

    fn = 'moon.bmp'

    with Image.open(fn) as img:
        gray = np.array(img.convert('L'))

    eq = hst_eq(gray)
    save_plt(eq, 'Equalized', 'moon_eq.png', rd)

    for mode in ['linear', 'quadratic']:
        srf = fit_surf(eq, mode)
        
        scl = scale_img(eq, srf)
        save_plt(scl, f'{mode.capitalize()} Scaled', f'moon_{mode}_scaled.png', rd)
        
        trc = trunc_img(eq, srf)
        save_plt(trc, f'{mode.capitalize()} Truncated', f'moon_{mode}_trunc.png', rd)

if __name__ == "__main__":
    main()