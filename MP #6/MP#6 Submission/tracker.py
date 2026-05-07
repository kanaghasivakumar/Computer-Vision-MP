import numpy as np
from PIL import Image, ImageDraw
import os
import glob

def load_frame(path):
    return np.array(Image.open(path).convert('RGB'), dtype=np.float64)

def to_gray(frame):
    return 0.299*frame[:,:,0] + 0.587*frame[:,:,1] + 0.114*frame[:,:,2]

def detect_face_region(frame):
    r, g, b = frame[:,:,0], frame[:,:,1], frame[:,:,2]
    skin = (
        (r > 95) & (g > 40) & (b > 20) &
        (r > g) & (r > b) &
        ((r - g) > 15) &
        (np.abs(r.astype(np.int32) - g.astype(np.int32)) > 15)
    )
    h, w = skin.shape
    mask = np.zeros_like(skin)
    y0, y1 = int(h * 0.05), int(h * 0.65)
    x0, x1 = int(w * 0.20), int(w * 0.80)
    mask[y0:y1, x0:x1] = True
    skin = skin & mask

    col_sum = skin.sum(axis=0)
    row_sum = skin.sum(axis=1)

    if col_sum.sum() == 0:
        return w // 2, h // 3, 60, 75

    xs = np.where(col_sum > col_sum.max() * 0.25)[0]
    ys = np.where(row_sum > row_sum.max() * 0.25)[0]

    cx = int((xs.min() + xs.max()) / 2)
    cy = int((ys.min() + ys.max()) / 2)
    box_w = max(40, xs.max() - xs.min())
    box_h = max(50, ys.max() - ys.min())
    box_w += box_w % 2
    box_h += box_h % 2

    hw, hh = box_w // 2, box_h // 2
    cx = int(np.clip(cx, hw, w - hw))
    cy = int(np.clip(cy, hh, h - hh))
    return cx, cy, box_w, box_h

def resize_template(template, new_hw, new_hh):
    return np.array(
        Image.fromarray(template.astype(np.float32)).resize(
            (new_hw*2, new_hh*2), Image.BILINEAR),
        dtype=np.float64
    )

def compute_ssd(patch, template):
    if patch.shape != template.shape:
        return np.inf
    d = patch - template
    return float(np.sum(d * d))

def compute_cc(patch, template):
    if patch.shape != template.shape:
        return -np.inf
    return float(np.sum(patch * template))

def compute_ncc(patch, template):
    if patch.shape != template.shape:
        return -np.inf
    p = patch - patch.mean()
    t = template - template.mean()
    denom = np.sqrt(np.sum(p*p) * np.sum(t*t))
    if denom < 1e-8:
        return 0.0
    return float(np.sum(p * t) / denom)

def score_patch(patch, template, method):
    if method == 'ssd':
        return compute_ssd(patch, template)
    elif method == 'cc':
        return compute_cc(patch, template)
    return compute_ncc(patch, template)

def is_better(s, best, method):
    return s < best if method == 'ssd' else s > best

def search_best_match(gray, template, cx, cy, hw, hh, radius, method):
    best_score = np.inf if method == 'ssd' else -np.inf
    best_cx, best_cy = cx, cy

    y0 = max(hh, cy - radius)
    y1 = min(gray.shape[0] - hh, cy + radius)
    x0 = max(hw, cx - radius)
    x1 = min(gray.shape[1] - hw, cx + radius)

    for ty in range(y0, y1 + 1):
        for tx in range(x0, x1 + 1):
            patch = gray[ty-hh:ty+hh, tx-hw:tx+hw]
            s = score_patch(patch, template, method)
            if is_better(s, best_score, method):
                best_score, best_cx, best_cy = s, tx, ty

    return best_cx, best_cy, hw, hh, best_score

def make_video_pillow(frame_dir, output_path, fps=25):
    paths = sorted(glob.glob(os.path.join(frame_dir, '*.png')))
    if not paths:
        return
    frames = [Image.open(p).convert('P', palette=Image.ADAPTIVE) for p in paths]
    frames[0].save(output_path, save_all=True, append_images=frames[1:],
                   duration=int(1000/fps), loop=0)
    print(f"  GIF saved: {output_path}")

def run_tracker(image_dir, method, search_radius=35):
    paths = sorted(
        glob.glob(os.path.join(image_dir, '*.jpg')) +
        glob.glob(os.path.join(image_dir, '*.png')) +
        glob.glob(os.path.join(image_dir, '*.bmp'))
    )

    out_frames = os.path.join(f'output_{method}', 'frames')
    out_video  = os.path.join(f'output_{method}', 'video')
    os.makedirs(out_frames, exist_ok=True)
    os.makedirs(out_video, exist_ok=True)

    first = load_frame(paths[0])
    cx, cy, box_w, box_h = detect_face_region(first)
    hw, hh = box_w // 2, box_h // 2
    print(f"[{method.upper()}] init: center=({cx},{cy}), box=({box_w}x{box_h})")

    gray0 = to_gray(first)
    working_template   = gray0[cy-hh:cy+hh, cx-hw:cx+hw].copy()
    reference_template = working_template.copy()
    reference_locked   = False
    clean_frame_count  = 0
    score_occluded     = 0

    for i, path in enumerate(paths):
        frame = load_frame(path)
        gray  = to_gray(frame)

        if i == 0:
            new_cx, new_cy, new_hw, new_hh, score = cx, cy, hw, hh, 0.0
            occ_type = None
        else:
            new_cx, new_cy, new_hw, new_hh, score = search_best_match(
                gray, working_template, cx, cy, hw, hh, search_radius, method)

            ref_scaled = resize_template(reference_template, new_hw, new_hh)
            patch = gray[new_cy-new_hh:new_cy+new_hh, new_cx-new_hw:new_cx+new_hw]
            ref_ncc = compute_ncc(patch, ref_scaled)
            occ_type = 'occluded' if ref_ncc < 0.60 else None

        if occ_type is not None:
            score_occluded += 1
            new_cx, new_cy = cx, cy
            new_hw, new_hh = hw, hh
        else:
            patch = gray[new_cy-new_hh:new_cy+new_hh, new_cx-new_hw:new_cx+new_hw]
            t = resize_template(working_template, new_hw, new_hh)
            if patch.shape == t.shape:
                working_template = 0.95 * t + 0.05 * patch

            clean_frame_count += 1
            if not reference_locked and clean_frame_count == 30:
                reference_template = resize_template(working_template, new_hw, new_hh).copy()
                reference_locked = True
                print(f"[{method.upper()}] reference template locked at frame {i}")

        cx, cy, hw, hh = new_cx, new_cy, new_hw, new_hh

        img_out = Image.fromarray(frame.astype(np.uint8))
        draw = ImageDraw.Draw(img_out)
        color = (255, 0, 0) if occ_type else (0, 255, 0)
        draw.ellipse([cx-hw, cy-hh, cx+hw, cy+hh], outline=color, width=2)
        img_out.save(os.path.join(out_frames, f'frame_{i:04d}.png'))

        if (i + 1) % 50 == 0:
            status = "OCCLUDED" if occ_type else "tracking"
            print(f"  frame {i+1:3d}/500 | ({cx:3d},{cy:3d}) | ellipse=({hw*2}x{hh*2}) | score={score:.4f} | {status}")

    print(f"[{method.upper()}] done — occluded: {score_occluded}/500\n")
    make_video_pillow(out_frames, os.path.join(out_video, f'tracking_{method}.gif'))

if __name__ == '__main__':
    for m in ['ssd', 'cc', 'ncc']:
        run_tracker('image_girl', method=m, search_radius=35)