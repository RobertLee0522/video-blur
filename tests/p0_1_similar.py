"""P0-1 相似窗戶誤鎖
V1-sim  : 兩扇同款窗框, 內部與周圍不同
V1-ident: A 與周圍 400px 完整複製成 B (外觀無法區分, 已知極限, 只記錄)
相機: 停在 A -> 快速平移到 B (A 離開, B 入鏡) -> 停在 B -> 快速平移回 A
"""
import os
import sys
import time

import numpy as np

from common import (OUT, T, camera_h, corner_err, env_info, iou, make_scene, poly_mask, project, read_frames,
                    rect_corners, render_view, save_json, smooth_path, state_of, visible_ratio, write_csv,
                    write_video, A_RECT, B_RECT)

FW, FH, FPS, N = 1920, 1080, 30, 420
VIEW_W, CY = 2400, 1100
CX_KEYS = [(0, 1450), (45, 1450), (105, 4550), (225, 4550), (285, 1450), (420, 1450)]
MIN_CONFS = [0.0, 0.3, 0.5, 0.7]
HOLD, PAD = 5, 0.05


def cam(i):
    cx = smooth_path(CX_KEYS, i)
    tilt = 120 * np.sin(i / 35.0)
    return camera_h(cx, CY, FW, FH, VIEW_W, tilt, roll=0.03 * np.sin(i / 50.0)), cx


def make_video(variant):
    path = os.path.join(OUT, "V1-%s.mp4" % variant)
    gt_path = path + ".gt.npz"
    if os.path.exists(path) and os.path.exists(gt_path):
        return path, dict(np.load(gt_path))
    scene = make_scene(variant)
    A, B = rect_corners(A_RECT), rect_corners(B_RECT)
    ga, gb = [], []

    def frames():
        prev_cx = None
        for i in range(N):
            H, cx = cam(i)
            ga.append(project(H, A))
            gb.append(project(H, B))
            speed = 0 if prev_cx is None else (cx - prev_cx) * FW / VIEW_W
            prev_cx = cx
            yield render_view(scene, H, FW, FH, speed)
    write_video(path, frames(), FW, FH, FPS)
    gt = {"A": np.array(ga), "B": np.array(gb)}
    np.savez(gt_path, **gt)
    return path, gt


def run(path, gt, variant, min_conf):
    eng = T.Engine()
    eng.min_conf, eng.hold_frames, eng.padding = min_conf, HOLD, PAD
    tgt = T.Target("A")
    eng.targets.append(tgt)

    shape = (FH, FW)
    cur = {"i": 0, "gray": None, "scale": 1.0}
    orb_events = []
    orig_recover = T.recover

    def rec(gray, ref, min_inliers=15):
        r = orig_recover(gray, ref, min_inliers)
        if r is not None:  # 記錄 ORB 找到的位置是 A 還是 B, 以及 NCC
            p = r / cur["scale"]
            m = poly_mask(shape, p)
            orb_events.append({
                "frame": cur["i"], "iou_A": round(iou(m, poly_mask(shape, gt["A"][cur["i"]])), 3),
                "iou_B": round(iou(m, poly_mask(shape, gt["B"][cur["i"]])), 3),
                "ncc": T.match_score(gray, ref, r), "min_inliers": min_inliers})
        return r
    T.recover = rec

    rows, prev, t_track = [], None, 0.0
    try:
        for i, frame in enumerate(read_frames(path)):
            gray, s = eng.to_gray(frame)
            cur.update(i=i, gray=gray, scale=s)
            if i == 0:
                tgt.set_keyframe(0, gt["A"][0], gray, s)
            t0 = time.time()
            (_, blur_poly, status), = eng.process(i, gray, prev, s)
            t_track += time.time() - t0
            prev = gray
            e = tgt.cache[i]
            st = state_of(status)
            gA, gB = poly_mask(shape, gt["A"][i]), poly_mask(shape, gt["B"][i])
            vis_a, vis_b = visible_ratio(shape, gt["A"][i]), visible_ratio(shape, gt["B"][i])
            mb = poly_mask(shape, T.expand_poly(np.asarray(blur_poly), PAD)) if blur_poly is not None else np.zeros(shape, bool)
            tp = poly_mask(shape, e["poly"]) if e["poly"] is not None else np.zeros(shape, bool)
            conf_b = None
            if vis_b >= 0.999:  # B 完整可見時, B 對 A 樣板的 NCC (可信度能否區分 A/B)
                conf_b = T.match_score(gray, tgt.refs[0], gt["B"][i] * s)
            rows.append({
                "frame": i, "state": st, "conf": None if e.get("conf") is None else round(e["conf"], 4),
                "lost": e["lost"], "poly": None if e["poly"] is None else np.round(e["poly"], 1).ravel().tolist(),
                "vis_A": round(vis_a, 4), "vis_B": round(vis_b, 4),
                "leak_A": round(float((gA & ~mb).sum()) / max(1, gA.sum()), 4) if vis_a >= 0.02 else None,
                "wrong_B": round(float((mb & gB).sum()) / max(1, gB.sum()), 4) if vis_b > 0 else 0.0,
                "iou_blur_A": round(iou(mb, gA), 4), "iou_A": round(iou(tp, gA), 4),
                "corner_err": round(corner_err(e["poly"], gt["A"][i]), 5) if (e["poly"] is not None and vis_a >= 0.999) else None,
                "conf_B_vs_template": None if conf_b is None else round(conf_b, 4),
            })
    finally:
        T.recover = orig_recover
    return rows, orb_events, t_track


def analyze(rows, orb_events, hold):
    def first(cond, start=0):
        return next((r["frame"] for r in rows[start:] if cond(r)), None)
    ev = {}
    ev["A_starts_leaving"] = first(lambda r: r["vis_A"] < 0.999)
    ev["A_fully_gone"] = first(lambda r: r["vis_A"] == 0)
    ev["B_enters"] = first(lambda r: r["vis_B"] > 0)
    ev["B_fully_visible"] = first(lambda r: r["vis_B"] >= 0.999)
    gone = ev["A_fully_gone"] or 0
    ev["A_returns_50pct"] = first(lambda r: r["vis_A"] >= 0.5, gone)
    ret = ev["A_returns_50pct"]
    ev["A_tracking_recovered"] = first(lambda r: r["state"] == "TRACKING" and r["iou_A"] >= 0.8, ret) if ret is not None else None
    ev["first_non_tracking"] = first(lambda r: r["state"] != "TRACKING")

    false_lock = [r for r in rows if r["state"] in ("TRACKING", "HOLD") and r["wrong_B"] >= 0.3 and r["iou_blur_A"] < 0.3]
    blurred_B = [r for r in rows if r["wrong_B"] > 0]
    good_conf = [r["conf"] for r in rows if r["state"] == "TRACKING" and r["iou_A"] >= 0.8 and r["conf"] is not None]
    conf_b = [r["conf_B_vs_template"] for r in rows if r["conf_B_vs_template"] is not None]
    lag = None if ret is None or ev["A_tracking_recovered"] is None else ev["A_tracking_recovered"] - ret

    post = []
    if ev["A_tracking_recovered"] is not None:
        k = ev["A_tracking_recovered"]
        post = rows[k:k + 30]
    post_ok = bool(post) and all((r["leak_A"] is None or r["leak_A"] <= 0.05) and r["iou_A"] >= 0.8 for r in post)

    # 離開時狀態序列: L..L+hold-1 HOLD, L+hold LOST_STOPPED
    seq_ok, seq = None, []
    L = ev["first_non_tracking"]
    if L is not None:
        seq = [rows[j]["state"] for j in range(L, min(len(rows), L + hold + 1))]
        expect = ["HOLD"] * hold + ["LOST_STOPPED"]
        seq_ok = seq == expect[:len(seq)]
    # A 可見 (>=50%) 但不是 TRACKING 的幀 (排除找回延遲期間)
    miss = [r["frame"] for r in rows if r["vis_A"] >= 0.5 and r["state"] != "TRACKING"
            and not (ret is not None and ret <= r["frame"] < (ev["A_tracking_recovered"] or 10 ** 9))]

    def dist(v):
        return None if not v else {"min": round(min(v), 3), "p5": round(float(np.percentile(v, 5)), 3),
                                   "median": round(float(np.median(v)), 3), "max": round(max(v), 3), "n": len(v)}
    orb_b = [o for o in orb_events if o["iou_B"] >= 0.5]
    return {
        "events": ev,
        "false_lock_frames": [r["frame"] for r in false_lock],
        "false_lock_detail": [{k: r[k] for k in ("frame", "state", "conf", "wrong_B", "iou_blur_A")} for r in false_lock[:20]],
        "frames_any_blur_on_B": len(blurred_B),
        "max_wrong_B": max([r["wrong_B"] for r in rows] or [0]),
        "conf_correct": dist(good_conf), "conf_B_vs_template": dist(conf_b),
        "recover_lag": lag, "post_recovery_30f_ok": post_ok,
        "post_recovery_max_leak": max([r["leak_A"] or 0 for r in post] or [None]) if post else None,
        "leave_state_sequence": seq, "leave_sequence_ok": seq_ok,
        "visible_but_not_tracking_frames": miss,
        "orb_found_B_count": len(orb_b),
        "orb_found_B_ncc": dist([o["ncc"] for o in orb_b if o["ncc"] is not None]),
        "orb_found_B_frames": [o["frame"] for o in orb_b][:30],
        "tracking_corner_err": dist([r["corner_err"] for r in rows if r["corner_err"] is not None and r["state"] == "TRACKING"]),
        "max_leak_A_while_vis50": max([r["leak_A"] for r in rows if r["leak_A"] is not None and r["vis_A"] >= 0.5] or [0]),
    }


def main():
    report = {"env": env_info(), "video": {"size": [FW, FH], "fps": FPS, "frames": N, "codec": "H.264 CRF12"},
              "params": {"TR": "TR-960", "hold_frames": HOLD, "padding": PAD}, "runs": {}}
    variants = sys.argv[1:] or ["sim", "twin", "ident"]
    old = os.path.join(OUT, "P0-1_report.json")
    if os.path.exists(old):  # 只跑部分變體時保留其他結果
        import json
        with open(old, encoding="utf-8") as f:
            report["runs"] = json.load(f).get("runs", {})
    for variant in variants:
        path, gt = make_video(variant)
        for mc in MIN_CONFS:
            t0 = time.time()
            rows, orb, t_track = run(path, gt, variant, mc)
            key = "V1-%s/min_conf=%.1f" % (variant, mc)
            write_csv(os.path.join(OUT, "P0-1_%s_minconf%.1f.csv" % (variant, mc)), rows)
            res = analyze(rows, orb, HOLD)
            res["track_ms_per_frame"] = round(t_track / len(rows) * 1000, 1)
            res["wall_s"] = round(time.time() - t0, 1)
            report["runs"][key] = res
            print(key, "false_lock=%d" % len(res["false_lock_frames"]), "lag=%s" % res["recover_lag"],
                  "orbB=%d" % res["orb_found_B_count"], "conf_ok=%s" % res["conf_correct"],
                  "confB=%s" % res["conf_B_vs_template"], flush=True)
    save_json(os.path.join(OUT, "P0-1_report.json"), report)


if __name__ == "__main__":
    main()
