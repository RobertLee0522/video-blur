"""P0-5 四點透視區域是否只模糊 polygon
1. 黑色遮擋 (無羽化): 塗黑區域 vs expand_poly(poly, padding) 的 IoU、多邊形外被改動像素、面積比
2. 高斯 / 馬賽克: 被改動像素是否都在「外擴多邊形 + 羽化半徑」內; 外接矩形內多邊形外的區域是否被動到
3. 高斯邊緣: 以 1px 棋盤格估計實際混合權重 alpha, 檢查「真實窗戶範圍 (未外擴)」內是否有模糊不完整 (alpha < 0.9) 的像素
4. 透視跟隨: 引用 P0-1 追蹤中 CornerErr
"""
import json
import os

import cv2
import numpy as np

from common import OUT, T, env_info, save_json

FW, FH = 1920, 1080
# 強烈透視的斜四邊形 (相機斜拍窗戶)
QUADS = {
    "oblique_left": np.float32([[520, 180], [1180, 330], [1160, 820], [540, 1000]]),
    "oblique_small": np.float32([[1500, 300], [1640, 340], [1635, 520], [1505, 590]]),
    "rotated": np.float32([[900, 150], [1500, 420], [1250, 950], [650, 680]]),
}
PADS = [0.0, 0.03, 0.05]


def mask_of(poly):
    m = np.zeros((FH, FW), np.uint8)
    cv2.fillPoly(m, [np.round(poly).astype(np.int32)], 1)
    return m.astype(bool)


def solid_test(poly, pad):
    bg = np.full((FH, FW, 3), (40, 180, 90), np.uint8)
    out = T.apply_blur(bg.copy(), poly, "solid", 50, pad)
    changed = np.any(out != bg, axis=2)
    exp = mask_of(T.expand_poly(poly, pad))
    inter, union = (changed & exp).sum(), (changed | exp).sum()
    exp_d = cv2.dilate(exp.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)  # 2 px
    x0, y0 = np.floor(T.expand_poly(poly, pad).min(axis=0)).astype(int)
    x1, y1 = np.ceil(T.expand_poly(poly, pad).max(axis=0)).astype(int)
    bbox = np.zeros((FH, FW), bool)
    bbox[max(y0, 0):y1, max(x0, 0):x1] = True
    area0 = abs(cv2.contourArea(poly))
    return {
        "iou_vs_expected": round(float(inter) / union, 4),
        "changed_outside_expected_plus2px": int((changed & ~exp_d).sum()),
        "bbox_minus_polygon_px": int((bbox & ~exp_d).sum()),
        "bbox_minus_polygon_changed_px": int((bbox & ~exp_d & changed).sum()),
        "area_ratio_measured": round(float(changed.sum()) / area0, 4),
        "area_ratio_analytic": round(abs(cv2.contourArea(T.expand_poly(poly, pad))) / area0, 4),
        "area_ratio_target": round((1 + pad) ** 2, 4),
    }


def soft_test(poly, pad, mode):
    rng = np.random.default_rng(0)
    bg = (rng.random((FH, FW, 3)) * 255).astype(np.uint8)
    out = T.apply_blur(bg.copy(), poly, mode, 50, pad)
    changed = np.any(out != bg, axis=2)
    p = T.expand_poly(poly, pad)
    size = max(np.ptp(p[:, 0]), np.ptp(p[:, 1]))
    feather_k = int(size * 0.02) * 2 + 1
    allowed = cv2.dilate(mask_of(p).astype(np.uint8), np.ones((feather_k + 2, feather_k + 2), np.uint8)).astype(bool)
    return {"feather_kernel_px": feather_k, "changed_outside_polygon_plus_feather": int((changed & ~allowed).sum()),
            "changed_px": int(changed.sum())}


def alpha_test(poly, pad):
    """1px 棋盤格 (0/255) 經強模糊後約為 127.5, 由輸出反推 alpha"""
    yy, xx = np.mgrid[0:FH, 0:FW]
    cb = (((xx + yy) % 2) * 255).astype(np.uint8)
    bg = np.dstack([cb] * 3)
    out = T.apply_blur(bg.copy(), poly, "gaussian", 100, pad).astype(np.float32)[..., 0]
    src = bg[..., 0].astype(np.float32)
    alpha = np.where(src > 127, (255 - out) / 127.5, out / 127.5)
    alpha = cv2.blur(alpha, (2, 2))  # 平均相鄰黑白格, 抵消模糊結果非精確 127.5 的誤差
    true_win = mask_of(poly)
    inner = cv2.erode(true_win.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    a = alpha[inner]
    edge = true_win & ~cv2.erode(true_win.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    return {"true_window_alpha_min": round(float(a.min()), 3), "true_window_alpha_p1": round(float(np.percentile(a, 1)), 3),
            "true_window_px_alpha_lt_0.9": int((a < 0.9).sum()), "true_window_px_alpha_lt_0.5": int((a < 0.5).sum()),
            "true_window_px": int(inner.sum()), "edge_band_alpha_mean": round(float(alpha[edge].mean()), 3)}


def main():
    report = {"env": env_info(), "frame": [FW, FH], "results": {}}
    for name, q in QUADS.items():
        for pad in PADS:
            key = "%s/padding=%d%%" % (name, round(pad * 100))
            r = {"solid": solid_test(q, pad), "gaussian": soft_test(q, pad, "gaussian"),
                 "pixelate": soft_test(q, pad, "pixelate"), "gaussian_alpha": alpha_test(q, pad)}
            report["results"][key] = r
            print(key, json.dumps(r, ensure_ascii=False), flush=True)
    p1 = os.path.join(OUT, "P0-1_report.json")
    if os.path.exists(p1):
        with open(p1, encoding="utf-8") as f:
            runs = json.load(f)["runs"]
        report["perspective_follow_from_P0-1"] = {k: v["tracking_corner_err"] for k, v in runs.items()}
    save_json(os.path.join(OUT, "P0-5_report.json"), report)


if __name__ == "__main__":
    main()
