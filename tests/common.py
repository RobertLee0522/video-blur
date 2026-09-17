"""測試共用工具: 合成場景、相機路徑、GT、影片寫出、逐幀指標.
不修改被測程式, 只透過 tracker / app 的公開介面呼叫."""
import csv
import json
import os
import platform
import shutil
import subprocess
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "tests", "out")
os.makedirs(OUT, exist_ok=True)

import tracker as T  # noqa: E402

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")
NOWIN = {"creationflags": 0x08000000} if sys.platform == "win32" else {}

STATE_OF = [("追蹤中", "TRACKING"), ("遺失(保留", "HOLD"), ("遺失-已停止模糊", "LOST_STOPPED"),
            ("未啟用", "INACTIVE"), ("尚未追蹤", "UNTRACKED")]


def state_of(status):
    for prefix, name in STATE_OF:
        if status.startswith(prefix):
            return name
    return "UNKNOWN:" + status


def env_info():
    def ver(cmd):
        try:
            return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **NOWIN).stdout.decode(
                "utf-8", "replace").splitlines()[0]
        except Exception:  # noqa
            return "N/A"
    return {
        "os": platform.platform(), "machine": platform.machine(), "processor": platform.processor(),
        "python": sys.version.split()[0], "opencv": cv2.__version__, "numpy": np.__version__,
        "ffmpeg": ver([FFMPEG, "-version"]) if FFMPEG else "N/A", "ffmpeg_path": FFMPEG,
    }


# ---------------------------------------------------------------- 場景
SCENE_W, SCENE_H = 6000, 2400
WIN_W, WIN_H = 500, 600
A_RECT = (1200, 800)   # 左上角
B_RECT = (4300, 800)


def rect_corners(xy, w=WIN_W, h=WIN_H):
    x, y = xy
    return np.float32([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])


def make_wall(seed):
    rng = np.random.default_rng(seed)
    wall = np.full((SCENE_H, SCENE_W, 3), (150, 160, 170), np.uint8)
    # 磚牆: 每塊磚顏色略不同, 提供追蹤特徵
    bh, bw = 60, 140
    for r in range(SCENE_H // bh + 1):
        off = (bw // 2) * (r % 2)
        for c in range(-1, SCENE_W // bw + 1):
            x0, y0 = c * bw + off, r * bh
            col = tuple(int(v) for v in rng.integers(90, 200, 3))
            cv2.rectangle(wall, (x0 + 3, y0 + 3), (x0 + bw - 3, y0 + bh - 3), col, -1)
    noise = rng.normal(0, 12, wall.shape)
    wall = np.clip(wall + noise, 0, 255).astype(np.uint8)
    for _ in range(400):  # 牆上雜物 (海報、污漬)
        c = tuple(int(v) for v in rng.integers(0, 255, 3))
        p = (int(rng.integers(0, SCENE_W)), int(rng.integers(0, SCENE_H)))
        cv2.circle(wall, p, int(rng.integers(6, 40)), c, -1)
    return wall


def draw_window(scene, xy, interior_seed):
    """同款窗框 (白框 + 十字窗格), 內部內容依 seed 不同"""
    x, y = xy
    rng = np.random.default_rng(interior_seed)
    inner = np.zeros((WIN_H, WIN_W, 3), np.uint8)
    inner[:] = rng.integers(40, 120, 3)
    for _ in range(60):
        c = tuple(int(v) for v in rng.integers(0, 255, 3))
        p1 = (int(rng.integers(0, WIN_W)), int(rng.integers(0, WIN_H)))
        p2 = (int(rng.integers(0, WIN_W)), int(rng.integers(0, WIN_H)))
        cv2.rectangle(inner, p1, p2, c, -1) if rng.random() < 0.5 else cv2.circle(inner, p1, int(rng.integers(10, 60)), c, -1)
    scene[y:y + WIN_H, x:x + WIN_W] = inner
    cv2.rectangle(scene, (x, y), (x + WIN_W, y + WIN_H), (245, 245, 245), 28)
    cv2.line(scene, (x + WIN_W // 2, y), (x + WIN_W // 2, y + WIN_H), (245, 245, 245), 18)
    cv2.line(scene, (x, y + WIN_H // 2), (x + WIN_W, y + WIN_H // 2), (245, 245, 245), 18)


def make_scene(variant, seed=7):
    """variant: 'sim' 窗框相同、內部與周圍不同; 'twin' 窗戶本體相同、周圍不同;
    'ident' A 與周圍大範圍像素完全複製到 B; 'single' 只有 A"""
    scene = make_wall(seed)
    draw_window(scene, A_RECT, 101)
    if variant == "sim":
        draw_window(scene, B_RECT, 202)
    elif variant == "twin":
        # 同型號窗戶、同樣窗簾 (窗戶本體像素相同), 但周圍牆面不同
        m = 16
        ax, ay = A_RECT
        bx, by = B_RECT
        scene[by - m:by + WIN_H + m, bx - m:bx + WIN_W + m] = scene[ay - m:ay + WIN_H + m, ax - m:ax + WIN_W + m]
    elif variant == "ident":
        m = 400  # 複製窗戶 + 周圍 400 px (遠大於樣板的 25% 外擴)
        ax, ay = A_RECT
        bx, by = B_RECT
        scene[ay - m:ay + WIN_H + m, bx - m:bx + WIN_W + m] = scene[ay - m:ay + WIN_H + m, ax - m:ax + WIN_W + m]
    return scene


# ---------------------------------------------------------------- 相機
def smooth_path(keys, i):
    """keys: [(frame, value)], 分段 smoothstep 內插"""
    if i <= keys[0][0]:
        return keys[0][1]
    for (f0, v0), (f1, v1) in zip(keys, keys[1:]):
        if f0 <= i <= f1:
            t = (i - f0) / float(max(1, f1 - f0))
            t = t * t * (3 - 2 * t)
            return v0 + (v1 - v0) * t
    return keys[-1][1]


def camera_h(cx, cy, fw, fh, view_w, tilt, roll=0.0):
    """回傳 scene -> frame 的 homography. tilt 造成左右透視 (相機左右轉), roll 小角度旋轉"""
    vw, vh = view_w, view_w * fh / float(fw)
    src = np.float32([[cx - vw / 2, cy - vh / 2 + tilt], [cx + vw / 2, cy - vh / 2 - tilt],
                      [cx + vw / 2, cy + vh / 2 + tilt], [cx - vw / 2, cy + vh / 2 - tilt]])
    if roll:
        c, s = np.cos(roll), np.sin(roll)
        src = (src - [cx, cy]).dot(np.float32([[c, -s], [s, c]]).T) + [cx, cy]
    dst = np.float32([[0, 0], [fw, 0], [fw, fh], [0, fh]])
    return cv2.getPerspectiveTransform(src.astype(np.float32), dst)


def render_view(scene, H, fw, fh, motion_px=0.0):
    f = cv2.warpPerspective(scene, H, (fw, fh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    k = int(abs(motion_px) * 0.5)
    if k >= 3:  # 快速平移的動態模糊
        ker = np.zeros((1, k), np.float32)
        ker[0, :] = 1.0 / k
        f = cv2.filter2D(f, -1, ker)
    return f


def project(H, corners):
    return cv2.perspectiveTransform(corners.reshape(-1, 1, 2).astype(np.float32), H).reshape(-1, 2)


# ---------------------------------------------------------------- 影片 IO
def write_video(path, frames, w, h, fps="30", vcodec=("-c:v", "libx264", "-crf", "12", "-preset", "fast"),
                audio_wav=None, extra=()):
    cmd = [FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "%dx%d" % (w, h),
           "-r", str(fps), "-i", "-"]
    if audio_wav:
        cmd += ["-i", audio_wav, "-map", "0:v:0", "-map", "1:a:0", "-c:a", "aac", "-b:a", "192k"]
    cmd += list(vcodec) + ["-pix_fmt", "yuv420p"] + list(extra) + [path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, **NOWIN)
    n = 0
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f).tobytes())
        n += 1
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError("ffmpeg 寫出失敗: %s" % path)
    return n


def ffprobe_json(path, *args):
    out = subprocess.run([FFPROBE, "-v", "error", "-print_format", "json"] + list(args) + [path],
                         stdout=subprocess.PIPE, **NOWIN).stdout
    return json.loads(out.decode("utf-8", "replace") or "{}")


def read_frames(path):
    cap = cv2.VideoCapture(path)
    while True:
        ok, f = cap.read()
        if not ok:
            break
        yield f
    cap.release()


# ---------------------------------------------------------------- 指標
def poly_mask(shape, poly):
    m = np.zeros(shape[:2], np.uint8)
    if poly is not None:
        cv2.fillPoly(m, [np.round(np.asarray(poly)).astype(np.int32)], 1)
    return m.astype(bool)


def visible_ratio(shape, poly):
    full = abs(cv2.contourArea(np.asarray(poly, np.float32)))
    return float(poly_mask(shape, poly).sum()) / full if full > 0 else 0.0


def iou(a, b):
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum()) / u if u else 0.0


def corner_err(poly, gt):
    diag = float(np.linalg.norm(gt[0] - gt[2]))
    return float(np.abs(np.linalg.norm(np.asarray(poly) - gt, axis=1)).max()) / diag


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
