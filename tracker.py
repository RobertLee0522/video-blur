"""追蹤與模糊核心 (與 UI 無關, 可單獨測試)

追蹤方式:
  - 平面目標 (例如窗戶) 用 KLT 光流 + Homography 逐幀追蹤, 相機轉向時四邊形會跟著透視變形
  - 追蹤遺失時, 用 ORB 特徵比對「框選當下的樣板」重新找回目標
每個目標以 keyframes 紀錄: {幀號: 多邊形} 代表從該幀開始追蹤, {幀號: STOP} 代表從該幀停止追蹤+取消模糊
"""
import cv2
import numpy as np

STOP = "STOP"
WORK_WIDTH = 960  # 追蹤時縮小到此寬度以加速


def to_work_gray(frame, work_width=WORK_WIDTH):
    """回傳 (縮小灰階圖, 縮放比例); work_width <= 0 代表用原始解析度"""
    h, w = frame.shape[:2]
    s = min(1.0, work_width / float(w)) if work_width and work_width > 0 else 1.0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if s < 1.0:
        gray = cv2.resize(gray, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
    return gray, s


def expand_poly(poly, ratio):
    c = poly.mean(axis=0)
    return (poly - c) * (1.0 + ratio) + c


def poly_area(poly):
    return abs(cv2.contourArea(poly.astype(np.float32)))


def poly_sane(poly, ref_area, img_shape, lo=0.15, hi=6.0):
    if poly is None or not np.all(np.isfinite(poly)):
        return False
    a = poly_area(poly)
    if ref_area <= 0 or not (lo < a / ref_area < hi):
        return False
    if len(poly) == 4 and not cv2.isContourConvex(poly.astype(np.float32).reshape(-1, 1, 2)):
        return False
    h, w = img_shape[:2]
    x0, y0 = poly.min(axis=0)
    x1, y1 = poly.max(axis=0)
    # 完全離開畫面視為遺失
    return x1 > 0 and y1 > 0 and x0 < w and y0 < h


def poly_inside(poly, img_shape, margin=2):
    h, w = img_shape[:2]
    return bool((poly.min(axis=0) >= -margin).all() and poly[:, 0].max() <= w + margin and poly[:, 1].max() <= h + margin)


def poly_mask(shape, poly, grow=0.0):
    m = np.zeros(shape[:2], np.uint8)
    p = expand_poly(poly, grow) if grow else poly
    cv2.fillPoly(m, [np.round(p).astype(np.int32)], 255)
    return m


class Target(object):
    _next_id = 1

    def __init__(self, name=None):
        self.id = Target._next_id
        Target._next_id += 1
        self.name = name or "目標 %d" % self.id
        self.blur_enabled = True
        self.keyframes = {}   # idx -> poly(原圖座標 Nx2 float32) 或 STOP
        self.refs = {}        # idx -> ORB 樣板 dict
        self.cache = {}       # idx -> dict(poly, lost, anchor, anchor_kf)

    # ---- 編輯 ----
    def set_keyframe(self, idx, poly, gray, scale):
        self.keyframes[idx] = np.asarray(poly, np.float32)
        self.refs[idx] = build_ref(gray, np.asarray(poly, np.float32) * scale)
        self.invalidate(idx)

    def stop_at(self, idx):
        self.keyframes[idx] = STOP
        self.refs.pop(idx, None)
        self.invalidate(idx)

    def invalidate(self, idx):
        for k in [k for k in self.cache if k >= idx]:
            del self.cache[k]

    def active_keyframe(self, idx):
        ks = [k for k in self.keyframes if k <= idx]
        if not ks:
            return None
        k = max(ks)
        return None if self.keyframes[k] is STOP else k

    def has_any_active(self):
        return any(v is not STOP for v in self.keyframes.values())


def build_ref(gray, poly_work):
    orb = cv2.ORB_create(1500)
    mask = poly_mask(gray.shape, poly_work, grow=0.25)
    kp, des = orb.detectAndCompute(gray, mask)
    pts = np.float32([k.pt for k in kp]) if kp else np.zeros((0, 2), np.float32)
    return {"pts": pts, "des": des, "poly": poly_work.copy(), "area": poly_area(poly_work),
            "gray": gray.copy(), "mask": mask}


def match_score(gray, ref, poly):
    """把框選當下的樣板依假設位置 poly 投影到目前畫面, 回傳可見區域的 NCC (-1~1), 無法評估回傳 None"""
    try:
        H = cv2.getPerspectiveTransform(ref["poly"][:4].astype(np.float32), poly[:4].astype(np.float32))
    except cv2.error:
        return None
    # 只在假設位置 (含外擴) 的外接矩形內計算, 高解析度時也很快
    h, w = gray.shape[:2]
    p = expand_poly(poly.astype(np.float32), 0.3)
    x0, y0 = np.maximum(np.floor(p.min(axis=0)).astype(int), 0)
    x1, y1 = np.minimum(np.ceil(p.max(axis=0)).astype(int), [w, h])
    if x1 - x0 < 10 or y1 - y0 < 10:
        return None
    H = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], np.float64).dot(H)
    size = (int(x1 - x0), int(y1 - y0))
    m = cv2.warpPerspective(ref["mask"], H, size, flags=cv2.INTER_NEAREST)
    m = cv2.erode(m, np.ones((5, 5), np.uint8))
    sel = m > 0
    if sel.sum() < 300:
        return None
    wref = cv2.warpPerspective(ref["gray"], H, size, flags=cv2.INTER_LINEAR)
    a = wref[sel].astype(np.float32)
    b = gray[y0:y1, x0:x1][sel].astype(np.float32)
    a -= a.mean()
    b -= b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 1e-6 else None


_orb_full = None
_matcher = None


def recover(gray, ref, min_inliers=15):
    """用 ORB 在整張圖找回樣板, 回傳工作座標多邊形或 None"""
    global _orb_full, _matcher
    if ref["des"] is None or len(ref["pts"]) < 8:
        return None
    if _orb_full is None:
        _orb_full = cv2.ORB_create(3000)
        _matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    kp, des = _orb_full.detectAndCompute(gray, None)
    if des is None or len(kp) < 8:
        return None
    matches = _matcher.knnMatch(ref["des"], des, k=2)
    good = [m[0] for m in matches if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]
    if len(good) < min_inliers:
        return None
    src = np.float32([ref["pts"][m.queryIdx] for m in good])
    dst = np.float32([kp[m.trainIdx].pt for m in good])
    H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    if H is None or inl is None:
        return None
    n = int(inl.sum())
    if n < min_inliers or n < 0.35 * len(good):
        return None
    poly = cv2.perspectiveTransform(ref["poly"].reshape(-1, 1, 2), H).reshape(-1, 2)
    return poly if poly_sane(poly, ref["area"], gray.shape, 0.2, 5.0) else None


def track_step(prev_gray, gray, poly, ref_area):
    """KLT + Homography 追蹤一幀, 回傳新多邊形 (工作座標) 或 None"""
    p0 = None
    for grow in (0.3, 1.0):
        mask = poly_mask(prev_gray.shape, poly, grow=grow)
        p0 = cv2.goodFeaturesToTrack(prev_gray, maxCorners=400, qualityLevel=0.005,
                                     minDistance=5, mask=mask, blockSize=5)
        if p0 is not None and len(p0) >= 10:
            break
    if p0 is None or len(p0) < 6:
        return None
    lk = dict(winSize=(21, 21), maxLevel=3,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **lk)
    if p1 is None:
        return None
    p0r, st2, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, p1, None, **lk)
    fb = np.linalg.norm((p0 - p0r).reshape(-1, 2), axis=1)
    ok = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < 1.5)
    a, b = p0.reshape(-1, 2)[ok], p1.reshape(-1, 2)[ok]
    if len(a) < 6:
        return None
    new = None
    if len(a) >= 10:
        H, inl = cv2.findHomography(a, b, cv2.RANSAC, 3.0)
        if H is not None and inl is not None and int(inl.sum()) >= 8:
            new = cv2.perspectiveTransform(poly.reshape(-1, 1, 2).astype(np.float32), H).reshape(-1, 2)
            if not poly_sane(new, poly_area(poly), gray.shape, 0.6, 1.6):
                new = None
    if new is None:  # 特徵少時退回相似變換
        M, inl = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if M is None or inl is None or int(inl.sum()) < 5:
            return None
        new = cv2.transform(poly.reshape(-1, 1, 2).astype(np.float32), M).reshape(-1, 2)
    if not poly_sane(new, ref_area, gray.shape):
        return None
    return new


class Engine(object):
    def __init__(self):
        self.targets = []
        self.hold_frames = 5         # 遺失後仍在最後位置模糊幾幀, 超過就自動停止模糊
        self.blur_mode = "gaussian"  # gaussian / pixelate / solid
        self.strength = 50           # 1~100
        self.padding = 0.05          # 模糊範圍向外擴張比例 (保留一點追蹤誤差的餘裕)
        self.recheck_every = 5       # 每幾幀用樣板校正一次漂移
        self.min_conf = 0.3          # 可信度 (與框選當下樣板的 NCC) 低於此值視為遺失, 0 = 不檢查
        self.work_width = WORK_WIDTH  # 追蹤運算寬度, 0 = 原始解析度; 變更後需重建樣板 (見 app)

    def to_gray(self, frame):
        return to_work_gray(frame, self.work_width)

    def process(self, idx, gray, prev_gray, scale):
        """計算第 idx 幀每個目標的狀態.
        gray / prev_gray 為工作尺寸灰階 (prev_gray 必須是 idx-1 幀, 沒有則 None).
        回傳 [(target, 要模糊的多邊形(原圖座標) or None, 狀態字串)]"""
        out = []
        for t in self.targets:
            poly, status = self._process_target(t, idx, gray, prev_gray, scale)
            out.append((t, poly, status))
        return out

    def _process_target(self, t, idx, gray, prev_gray, scale):
        kf = t.active_keyframe(idx)
        if kf is None:
            return None, "未啟用"
        e = t.cache.get(idx)
        if e is None:
            if kf == idx:
                p = t.keyframes[idx]
                e = {"poly": p, "lost": 0, "anchor": p, "kf": kf, "conf": 1.0}
            else:
                pe = t.cache.get(idx - 1)
                if pe is None or prev_gray is None or pe["kf"] != kf:
                    return None, "尚未追蹤到此幀"
                e = self._track(t, idx, kf, pe, gray, prev_gray, scale)
            t.cache[idx] = e
        conf = e.get("conf")
        ctxt = "" if conf is None else " %d%%" % round(max(conf, 0) * 100)
        if e["lost"] == 0:
            return e["poly"], "追蹤中" + ctxt
        if e["lost"] <= self.hold_frames:
            return e["anchor"], "遺失(保留模糊 %d)" % e["lost"]
        return None, "遺失-已停止模糊"

    def _track(self, t, idx, kf, pe, gray, prev_gray, scale):
        ref = t.refs[kf]
        new, score = None, None
        if pe["lost"] == 0:
            new = track_step(prev_gray, gray, pe["poly"] * scale, ref["area"])
            if new is not None:
                score = match_score(gray, ref, new)
                if self.min_conf > 0 and score is not None and score < self.min_conf:
                    new = None  # 位置可疑 (可能追到別的東西), 寧可視為遺失
        edge = new is not None and not poly_inside(new, gray.shape)
        if new is None or edge or (idx - kf) % self.recheck_every == 0:
            # 遺失時找回; 追蹤中也定期 (目標在畫面邊緣時每幀) 與樣板比對修正漂移.
            # 兩個候選用樣板相似度 (NCC) 決定採用哪一個
            r = recover(gray, ref, min_inliers=8 if new is None else 15)
            if r is not None:
                sr = match_score(gray, ref, r)
                if new is None:
                    # 找回時要求較高可信度, 避免認成另一扇相似的窗戶
                    if sr is not None and sr > max(0.5, self.min_conf):
                        new, score = r, sr
                elif sr is not None and (score is None or sr > score + 0.02):
                    new, score = r, sr
        if new is not None:
            p = (new / scale).astype(np.float32)
            return {"poly": p, "lost": 0, "anchor": p, "kf": kf, "conf": score}
        return {"poly": None, "lost": pe["lost"] + 1, "anchor": pe["anchor"], "kf": kf, "conf": None}

    def render(self, frame, results):
        for t, poly, _ in results:
            if poly is not None and t.blur_enabled:
                apply_blur(frame, poly, self.blur_mode, self.strength, self.padding)
        return frame


def apply_blur(frame, poly, mode="gaussian", strength=50, padding=0.08):
    h, w = frame.shape[:2]
    p = expand_poly(np.asarray(poly, np.float32), padding)
    x0, y0 = np.floor(p.min(axis=0)).astype(int)
    x1, y1 = np.ceil(p.max(axis=0)).astype(int)
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return frame
    roi = frame[y0:y1, x0:x1]
    rh, rw = roi.shape[:2]
    size = max(rw, rh)
    s = max(1, min(100, int(strength)))
    if mode == "pixelate":
        block = max(2, int(size * s / 400.0))
        small = cv2.resize(roi, (max(1, rw // block), max(1, rh // block)), interpolation=cv2.INTER_AREA)
        eff = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)
    elif mode == "solid":
        eff = np.zeros_like(roi)
    else:
        f = max(1.0, size / 160.0)  # 先縮小再模糊, 大區域也很快
        small = cv2.resize(roi, (max(1, int(rw / f)), max(1, int(rh / f))), interpolation=cv2.INTER_AREA)
        k = int(max(small.shape[:2]) * s / 250.0) * 2 + 1
        small = cv2.GaussianBlur(small, (max(3, k), max(3, k)), 0)
        small = cv2.GaussianBlur(small, (max(3, k), max(3, k)), 0)
        eff = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_LINEAR)
    mask = np.zeros((rh, rw), np.uint8)
    cv2.fillPoly(mask, [np.round(p - [x0, y0]).astype(np.int32)], 255)
    if mode != "solid":
        feather = max(1, int(size * 0.02)) * 2 + 1
        mask = cv2.GaussianBlur(mask, (feather, feather), 0)
    a = (mask.astype(np.float32) / 255.0)[..., None]
    frame[y0:y1, x0:x1] = (eff * a + roi * (1.0 - a)).astype(np.uint8)
    return frame
