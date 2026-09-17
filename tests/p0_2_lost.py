"""P0-2 遺失 -> HOLD -> LOST_STOPPED 是否真的完全不模糊
V6-occlude (只有窗戶 A):
  f60-62   遮擋 A 左半 (50%) 3 幀                 情境 B
  f120-149 完全遮擋 A (100%) 30 幀                情境 C
  f200-380 相機平移離開 (A 消失約 120 幀) 再回來   情境 C/D
  f470-472 甩鏡, A 離開畫面 3 幀                   情境 A
像素級檢查: 用 engine.render 實際繪製, 比對與原畫面差異的像素.
"""
import os
import tkinter as tk

import cv2
import numpy as np

from common import (OUT, T, camera_h, env_info, iou, make_scene, poly_mask, project, read_frames, rect_corners,
                    render_view, save_json, smooth_path, state_of, visible_ratio, write_csv, write_video, A_RECT)

FW, FH, FPS, N = 1920, 1080, 30, 520
VIEW_W, CY = 2400, 1100
PAN_KEYS = [(0, 1450), (200, 1450), (260, 4550), (320, 4550), (380, 1450), (520, 1450)]
OCC_HALF = range(60, 63)
OCC_FULL = range(120, 150)
WHIP = range(470, 473)
PAD = 0.05


def cam(i):
    cx = smooth_path(PAN_KEYS, i) + 60 * np.sin(i / 40.0)
    if i in WHIP:
        cx = 4550
    return camera_h(cx, CY, FW, FH, VIEW_W, 60 * np.sin(i / 45.0)), cx


def occluder_rect(i, a_poly):
    x0, y0 = a_poly.min(axis=0)
    x1, y1 = a_poly.max(axis=0)
    if i in OCC_HALF:
        return int(x0) - 10, int(y0) - 10, int((x0 + x1) / 2), int(y1) + 10
    if i in OCC_FULL:
        w, h = x1 - x0, y1 - y0
        return int(x0 - 0.15 * w), int(y0 - 0.15 * h), int(x1 + 0.15 * w), int(y1 + 0.15 * h)
    return None


def make_video():
    path = os.path.join(OUT, "V6-occlude.mp4")
    gt_path = path + ".gt.npz"
    if os.path.exists(path) and os.path.exists(gt_path):
        return path, dict(np.load(gt_path))
    scene = make_scene("single")
    A = rect_corners(A_RECT)
    rng = np.random.default_rng(3)
    patch = np.clip(rng.normal(70, 25, (FH, FW, 3)), 0, 255).astype(np.uint8)  # 遮擋物 (例如路人)
    ga, occ = [], []

    def frames():
        prev = None
        for i in range(N):
            H, cx = cam(i)
            pa = project(H, A)
            ga.append(pa)
            f = render_view(scene, H, FW, FH, 0 if prev is None or i in WHIP or (i - 1) in WHIP else (cx - prev) * FW / VIEW_W)
            prev = cx
            r = occluder_rect(i, pa)
            occ.append(r if r else (0, 0, 0, 0))
            if r:
                x0, y0, x1, y1 = max(r[0], 0), max(r[1], 0), min(r[2], FW), min(r[3], FH)
                f[y0:y1, x0:x1] = patch[y0:y1, x0:x1]
            yield f
    write_video(path, frames(), FW, FH, FPS)
    gt = {"A": np.array(ga), "occ": np.array(occ)}
    np.savez(gt_path, **gt)
    return path, gt


def run(path, gt, hold):
    eng = T.Engine()
    eng.hold_frames, eng.padding = hold, PAD
    tgt = T.Target("A")
    eng.targets.append(tgt)
    shape = (FH, FW)
    rows, prev = [], None
    for i, frame in enumerate(read_frames(path)):
        gray, s = eng.to_gray(frame)
        if i == 0:
            tgt.set_keyframe(0, gt["A"][0], gray, s)
        res = eng.process(i, gray, prev, s)
        prev = gray
        _, blur_poly, status = res[0]
        e = tgt.cache[i]
        out = eng.render(frame.copy(), res)
        changed = np.any(out != frame, axis=2)
        gA = poly_mask(shape, gt["A"][i])
        x0, y0, x1, y1 = gt["occ"][i]
        occm = np.zeros(shape, bool)
        occm[max(y0, 0):max(y1, 0), max(x0, 0):max(x1, 0)] = True
        unocc = gA & ~occm
        mb = poly_mask(shape, T.expand_poly(np.asarray(blur_poly), PAD)) if blur_poly is not None else np.zeros(shape, bool)
        outside = 0
        if blur_poly is not None:  # 改動像素是否都在 (外擴多邊形 + 羽化) 範圍內
            p = T.expand_poly(np.asarray(blur_poly), PAD)
            feather = int(max(np.ptp(p[:, 0]), np.ptp(p[:, 1])) * 0.02) * 2 + 3
            allowed = cv2.dilate(mb.astype(np.uint8), np.ones((feather, feather), np.uint8)).astype(bool)
            outside = int((changed & ~allowed).sum())
        rows.append({
            "frame": i, "state": state_of(status), "status": status, "lost": e["lost"],
            "conf": None if e.get("conf") is None else round(e["conf"], 4),
            "vis_A": round(visible_ratio(shape, gt["A"][i]), 4),
            "occluded": "half" if i in OCC_HALF else "full" if i in OCC_FULL else "",
            "changed_px": int(changed.sum()), "changed_outside_allowed_px": outside,
            "leak_unoccluded_A": round(float((unocc & ~mb).sum()) / max(1, unocc.sum()), 4) if unocc.sum() > 500 else None,
            "iou_track_A": round(iou(poly_mask(shape, e["poly"]), gA), 4) if e["poly"] is not None else None,
            "iou_anchor_A": round(iou(poly_mask(shape, e["anchor"]), gA), 4),
        })
    return rows


def segments(rows):
    """連續非 TRACKING 的區段 [(L, end)]"""
    segs, L = [], None
    for r in rows + [{"frame": len(rows), "state": "TRACKING"}]:
        if r["state"] != "TRACKING" and L is None:
            L = r["frame"]
        elif r["state"] == "TRACKING" and L is not None:
            segs.append((L, r["frame"] - 1))
            L = None
    return segs


def analyze(rows, hold):
    segs = segments(rows)
    seg_checks = []
    for L, end in segs:
        bad = []
        for j in range(L, end + 1):
            r = rows[j]
            exp = "HOLD" if (j - L) < hold else "LOST_STOPPED"
            if r["state"] != exp or r["lost"] != j - L + 1:
                bad.append({"frame": j, "state": r["state"], "lost": r["lost"], "expected": exp})
        stopped = [rows[j] for j in range(L, end + 1) if rows[j]["state"] == "LOST_STOPPED"]
        holds = [rows[j] for j in range(L, end + 1) if rows[j]["state"] == "HOLD"]
        seg_checks.append({
            "L": L, "end": end, "length": end - L + 1, "state_mismatch": bad[:10],
            "lost_stopped_frames": len(stopped),
            "lost_stopped_changed_px_max": max([r["changed_px"] for r in stopped] or [0]),
            "hold_frames": len(holds), "hold_all_blurred": all(r["changed_px"] > 0 for r in holds),
            "hold_anchor_iou_min": min([r["iou_anchor_A"] for r in holds] or [None]) if holds else None,
            "recover_frame": end + 1 if end + 1 < len(rows) else None,
        })

    def lag_after(start):
        ret = next((r["frame"] for r in rows[start:] if r["vis_A"] >= 0.5 and not r["occluded"]), None)
        rec = next((r["frame"] for r in rows[ret:] if r["state"] == "TRACKING" and (r["iou_track_A"] or 0) >= 0.8), None) if ret is not None else None
        return ret, rec, None if rec is None else rec - ret

    half = [rows[i] for i in OCC_HALF]
    full = [rows[i] for i in OCC_FULL]
    pan_gone = next(r["frame"] for r in rows[200:] if r["vis_A"] == 0)
    return {
        "segments": seg_checks,
        "scenario_A_whip": {"states": [rows[i]["state"] for i in range(WHIP.start - 1, WHIP.stop + 3)],
                            "blurred_during": [rows[i]["changed_px"] > 0 for i in WHIP],
                            "return_lag": lag_after(WHIP.stop)},
        "scenario_B_half_occlusion": {"states": [r["state"] for r in half],
                                      "max_leak_unoccluded": max([r["leak_unoccluded_A"] or 0 for r in half]),
                                      "after_3f_states": [rows[i]["state"] for i in range(63, 68)]},
        "scenario_C_full_occlusion": {"states": [r["state"] for r in full],
                                      "after_states": [rows[i]["state"] for i in range(150, 160)],
                                      "return_lag": lag_after(150)},
        "scenario_C_pan_out": {"A_gone_frame": pan_gone, "return_lag": lag_after(pan_gone + 1)},
        "lost_stopped_total": sum(1 for r in rows if r["state"] == "LOST_STOPPED"),
        "lost_stopped_any_blur": sum(1 for r in rows if r["state"] == "LOST_STOPPED" and r["changed_px"] > 0),
        "tracking_wrong_position_frames": [r["frame"] for r in rows if r["state"] == "TRACKING" and r["iou_track_A"] is not None
                                           and r["iou_track_A"] < 0.5 and r["vis_A"] >= 0.5 and not r["occluded"]],
        "tracking_on_occluder_frames": [r["frame"] for r in rows if r["occluded"] == "full" and r["state"] == "TRACKING"],
        "changed_outside_allowed_max": max(r["changed_outside_allowed_px"] for r in rows),
    }


def ui_check(path, gt, rows5):
    """用真正的 App 在 L+2 / L+6 檢查畫布框線與清單狀態 (不截圖, 直接讀畫布物件)"""
    import app as A
    segs = [s for s in segments(rows5) if s[1] - s[0] + 1 >= 8]
    if not segs:
        return {"error": "no long lost segment"}
    root = tk.Tk()
    a = A.App(root)
    root.update()
    a.open_video(path)
    t = T.Target("A")
    a.engine.targets.append(t)
    t.set_keyframe(0, gt["A"][0], a.cur_gray, a.scale)
    a.refresh_tree()
    st = {"i": 0, "done": False, "cancel": False, "error": None, "msg": ""}
    a._batch_worker(path, None, st)
    out = {"analysis_error": st["error"]}
    for L, end in segs:
        for off in (2, 6):
            f = L + off
            a.cur_idx = -1
            a.goto(f)
            a.tree.selection_set(str(t.id))
            root.update()
            items = []
            for it in a.canvas.find_all():
                kind = a.canvas.type(it)
                if kind == "polygon":
                    items.append({"type": kind, "outline": a.canvas.itemcget(it, "outline"), "dash": a.canvas.itemcget(it, "dash")})
                elif kind == "text":
                    items.append({"type": kind, "text": a.canvas.itemcget(it, "text"), "fill": a.canvas.itemcget(it, "fill")})
            out["L=%d+%d" % (L, off)] = {"status": a.results[0][2], "tree": a.tree.item(str(t.id), "values")[1],
                                         "cache_state_matches_headless": state_of(a.results[0][2]) == rows5[f]["state"],
                                         "canvas": items}
    root.destroy()
    return out


def main():
    import sys
    path, gt = make_video()
    if "--ui-only" in sys.argv:  # 只補跑 UI 檢查, 其他沿用既有報告
        import json
        with open(os.path.join(OUT, "P0-2_report.json"), encoding="utf-8") as f:
            report = json.load(f)
        report["ui_check_hold5"] = ui_check(path, gt, run(path, gt, 5))
        save_json(os.path.join(OUT, "P0-2_report.json"), report)
        print(json.dumps(report["ui_check_hold5"], ensure_ascii=False)[:3000])
        return
    report = {"env": env_info(), "video": {"size": [FW, FH], "fps": FPS, "frames": N}, "params": {"TR": "TR-960", "min_conf": 0.3, "padding": PAD}, "runs": {}}
    rows5 = None
    for hold in (0, 5, 15):
        rows = run(path, gt, hold)
        write_csv(os.path.join(OUT, "P0-2_hold%d.csv" % hold), rows)
        res = analyze(rows, hold)
        report["runs"]["hold=%d" % hold] = res
        if hold == 5:
            rows5 = rows
        print("hold", hold, "segments", [(s["L"], s["end"], len(s["state_mismatch"]), s["lost_stopped_changed_px_max"]) for s in res["segments"]],
              "stopped_blur", res["lost_stopped_any_blur"], "outside", res["changed_outside_allowed_max"], flush=True)
        save_json(os.path.join(OUT, "P0-2_report.json"), report)
    report["ui_check_hold5"] = ui_check(path, gt, rows5)
    save_json(os.path.join(OUT, "P0-2_report.json"), report)


if __name__ == "__main__":
    main()
