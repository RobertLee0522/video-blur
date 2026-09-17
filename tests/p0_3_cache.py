"""P0-3 分析快取 -> 匯出是否真的避免重新追蹤
E1 分析後匯出: 計數 track_step / recover 呼叫次數, 量 to_gray 耗時
E2 分析->匯出 vs 直接匯出: 每幀 poly 與輸出畫面是否一致
E3 seek 對齊: seek(k) 讀到的畫面 vs 從 0 依序讀到第 k 幀 (H.264 長 GOP+B 幀 / HEVC 10-bit / 可變幀率)
E4 各種操作對快取的影響
E5 分析範圍 [300, 900) 與匯出時範圍外不模糊; CAP_PROP_FRAME_COUNT 準確度
"""
import hashlib
import os
import subprocess
import time
import tkinter as tk

import cv2
import numpy as np

from common import (FFMPEG, NOWIN, OUT, T, camera_h, env_info, make_scene, project, read_frames, rect_corners,
                    render_view, save_json, state_of, write_video, A_RECT)

FW, FH, FPS, N = 1280, 720, 30, 1500
VIEW_W, CY = 2400, 1100
K_START, K_STOP = 300, 900


def cam(i):
    cx = 1450 + 250 * np.sin(i / 90.0)
    return camera_h(cx, CY + 80 * np.sin(i / 70.0), FW, FH, VIEW_W, 90 * np.sin(i / 60.0))


def gen_frames(scene, n, cam_fn, gt=None):
    A = rect_corners(A_RECT)
    for i in range(n):
        H = cam_fn(i)
        if gt is not None:
            gt.append(project(H, A))
        yield render_view(scene, H, FW, FH)


def make_videos():
    scene = make_scene("single")
    vids = {}
    p = os.path.join(OUT, "V-cache_h264_gop250_bf3.mp4")
    gt_path = p + ".gt.npy"
    wav = os.path.join(OUT, "tone.wav")
    if not os.path.exists(wav):
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=60", wav], **NOWIN)
    if not (os.path.exists(p) and os.path.exists(gt_path)):
        gt = []
        write_video(p, gen_frames(scene, N, cam, gt), FW, FH, FPS,
                    vcodec=("-c:v", "libx264", "-crf", "16", "-preset", "fast", "-g", "250", "-bf", "3"),
                    audio_wav=wav, extra=("-shortest",))
        np.save(gt_path, np.array(gt))
    vids["H.264 GOP250 B3"] = p

    p2 = os.path.join(OUT, "V-cache_hevc10_hdrtag.mp4")
    if not os.path.exists(p2):
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", p, "-frames:v", "600", "-c:v", "libx265",
                        "-pix_fmt", "yuv420p10le", "-x265-params", "keyint=250:bframes=4:log-level=error",
                        "-color_primaries", "bt2020", "-color_trc", "smpte2084", "-colorspace", "bt2020nc",
                        "-tag:v", "hvc1", "-an", p2], **NOWIN)
    vids["HEVC 10-bit HDR-tag"] = p2

    p3 = os.path.join(OUT, "V-cache_vfr_30_60_24.mp4")
    if not os.path.exists(p3):
        segs = []
        for j, (fps, n) in enumerate([(30, 300), (60, 300), (24, 240)]):
            sp = os.path.join(OUT, "vfr_seg%d.mp4" % j)
            write_video(sp, gen_frames(scene, n, lambda i, j=j: cam(i + j * 400)), FW, FH, fps,
                        vcodec=("-c:v", "libx264", "-crf", "16", "-preset", "fast", "-g", "120", "-bf", "2"),
                        extra=("-video_track_timescale", "90000"))
            segs.append(sp)
        lst = os.path.join(OUT, "vfr_list.txt")
        with open(lst, "w") as f:
            f.writelines("file '%s'\n" % os.path.basename(s) for s in segs)
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", p3],
                       cwd=OUT, **NOWIN)
    vids["VFR 30/60/24 fps"] = p3
    return vids, np.load(gt_path)


def md5(f):
    return hashlib.md5(f.tobytes()).hexdigest()


def e3_seek(vids):
    out = {}
    rng = np.random.default_rng(0)
    for name, p in vids.items():
        seq = [md5(f) for f in read_frames(p)]
        cap = cv2.VideoCapture(p)
        count_prop = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_prop = cap.get(cv2.CAP_PROP_FPS)
        ks = sorted(rng.choice(len(seq), size=min(200, len(seq)), replace=False).tolist())
        bad = []
        for k in ks:
            cap.set(cv2.CAP_PROP_POS_FRAMES, k)
            ok, f = cap.read()
            h = md5(f) if ok else None
            if h != seq[k]:
                near = [d for d in range(-15, 16) if 0 <= k + d < len(seq) and seq[k + d] == h]
                bad.append({"k": k, "read_ok": ok, "matches_frame_offset": near[0] if near else None})
        cap.release()
        out[name] = {"frames_sequential": len(seq), "CAP_PROP_FRAME_COUNT": count_prop, "CAP_PROP_FPS": round(fps_prop, 4),
                     "sampled": len(ks), "mismatch": len(bad), "mismatch_detail": bad[:15],
                     "offsets_seen": sorted(set(b["matches_frame_offset"] for b in bad if b["matches_frame_offset"] is not None))}
        print("E3", name, out[name]["mismatch"], "/", len(ks), out[name]["offsets_seen"], flush=True)
    return out


class Counter(object):
    def __init__(self):
        self.track = self.recover = 0
        self.gray_s = 0.0
        self.gray_n = 0
        self._t, self._r = T.track_step, T.recover

    def install(self, engine):
        c = self

        def ts(*a, **k):
            c.track += 1
            return c._t(*a, **k)

        def rc(*a, **k):
            c.recover += 1
            return c._r(*a, **k)
        T.track_step, T.recover = ts, rc
        orig = engine.to_gray

        def tg(frame):
            t0 = time.time()
            r = orig(frame)
            c.gray_s += time.time() - t0
            c.gray_n += 1
            return r
        engine.to_gray = tg
        self._engine, self._orig_gray = engine, orig

    def uninstall(self):
        T.track_step, T.recover = self._t, self._r
        self._engine.to_gray = self._orig_gray


def new_app(path):
    import app as A
    root = tk.Tk()
    a = A.App(root)
    root.update()
    a.open_video(path)
    return root, a


def add_target(a, gt):
    a.goto(K_START)
    t = T.Target("A")
    a.engine.targets.append(t)
    t.set_keyframe(K_START, gt[K_START], a.cur_gray, a.scale)
    t.stop_at(K_STOP)
    a.refresh_tree()
    a.tree.selection_set(str(t.id))
    return t


def batch(a, out_path=None, encoder="libx264", record=None):
    st = {"i": 0, "done": False, "cancel": False, "error": None, "msg": "", "encoder": encoder}
    orig = a.engine.process
    if record is not None:
        def proc(idx, *args):
            res = orig(idx, *args)
            record[idx] = [None if p is None else np.asarray(p).copy() for _, p, _ in res]
            return res
        a.engine.process = proc
    t0 = time.time()
    try:
        a._batch_worker(a.path, out_path, st)
    finally:
        a.engine.process = orig
    st["wall_s"] = round(time.time() - t0, 2)
    return st


def cache_summary(t):
    ks = sorted(t.cache)
    return {"len": len(ks), "min": ks[0] if ks else None, "max": ks[-1] if ks else None}


def frames_psnr(p1, p2):
    worst, n, diff_frames = float("inf"), 0, 0
    for f1, f2 in zip(read_frames(p1), read_frames(p2)):
        n += 1
        if not np.array_equal(f1, f2):
            diff_frames += 1
            worst = min(worst, cv2.PSNR(f1, f2))
    return {"frames": n, "frames_different": diff_frames, "min_psnr": None if worst == float("inf") else round(worst, 2)}


def main():
    vids, gt = make_videos()
    path = vids["H.264 GOP250 B3"]
    report = {"env": env_info(), "videos": {k: os.path.basename(v) for k, v in vids.items()},
              "params": {"TR": "TR-960", "keyframe": K_START, "stop": K_STOP, "min_conf": 0.3, "hold": 5}}

    # ---- E1 + E5
    root, a = new_app(path)
    t = add_target(a, gt)
    report["E5_analysis_range"] = list(a.analysis_range())
    rec_an = {}
    st = batch(a, None, record=rec_an)
    report["E5_analysis"] = {"error": st["error"], "progress_total": st.get("total"), "progress_done": st["i"],
                             "processed_frames": [min(rec_an), max(rec_an)] if rec_an else None, "n_processed": len(rec_an),
                             "cache": cache_summary(t), "wall_s": st["wall_s"]}
    cache_after_analysis = {k: (None if v["poly"] is None else v["poly"].copy()) for k, v in t.cache.items()}
    c = Counter()
    c.install(a.engine)
    rec_ex = {}
    out1 = os.path.join(OUT, "P0-3_export_after_analysis.mp4")
    st = batch(a, out1, record=rec_ex)
    c.uninstall()
    outside_blur = [k for k, v in rec_ex.items() if (k < K_START or k >= K_STOP) and any(p is not None for p in v)]
    inside_none = [k for k, v in rec_ex.items() if K_START <= k < K_STOP and all(p is None for p in v)]
    report["E1_export_after_analysis"] = {
        "error": st["error"], "wall_s": st["wall_s"], "frames": len(rec_ex),
        "track_step_calls": c.track, "recover_calls": c.recover,
        "to_gray_calls": c.gray_n, "to_gray_total_s": round(c.gray_s, 2),
        "to_gray_on_cached_or_inactive_s": round(c.gray_s, 2),
        "cache_changed_during_export": sorted(set(t.cache) ^ set(cache_after_analysis))[:10],
    }
    report["E5_export"] = {"frames_outside_range_with_blur": outside_blur[:20], "n_outside_blur": len(outside_blur),
                           "frames_inside_range_without_blur_poly": inside_none[:20], "n_inside_none": len(inside_none)}
    print("E1", report["E1_export_after_analysis"], flush=True)


    # ---- E4 快取清除規則
    import app as A
    e4 = []

    def reanalyze():
        batch(a, None)

    def op(name, fn, expect):
        reanalyze()
        tt = a.engine.targets[0] if a.engine.targets else None
        before = cache_summary(tt) if tt else None
        fn()
        root.update()
        after = [cache_summary(x) for x in a.engine.targets]
        e4.append({"op": name, "cur_idx": a.cur_idx, "before": before, "after": after, "expected": expect})
        print("E4", name, before, "->", after, flush=True)

    def set_var(var, val):
        def f():
            var.set(val)
            a._apply_settings()
        return f
    op("hold_frames 5->9", set_var(a.hold_var, 9), "不清除")
    op("padding 5%->3%", set_var(a.pad_var, 3), "不清除")
    op("strength 50->70", set_var(a.strength_var, 70), "不清除")
    op("mode -> 馬賽克", set_var(a.mode_var, "馬賽克"), "不清除")
    op("blur toggle", a.toggle_blur, "不清除")

    def redraw():
        a.goto(600)
        a.redraw_target = a.engine.targets[0]
        a._commit(a.engine.targets[0].cache[600]["poly"].copy())
    op("重新框選 @600", redraw, ">=600 清除 (之後重算當前幀 600)")

    def stop():
        a.goto(700)
        a.tree.selection_set(str(a.engine.targets[0].id))
        a.stop_selected()
    op("STOP @700", stop, ">=700 清除")

    def resume():
        a.goto(760)
        a.tree.selection_set(str(a.engine.targets[0].id))
        a.resume_selected()
    op("恢復 (移除 STOP 700) @760", resume, ">=700 清除")
    op("min_conf 30->45", set_var(a.conf_var, 45), "全部清除")

    def tr():
        a.work_var.set(A.WORK_WIDTHS[1][0])
        a.change_work_width()
    op("TR-960 -> TR-1440", tr, "全部清除 + 重建樣板")

    from tkinter import filedialog
    proj = os.path.join(OUT, "P0-3_proj.json")
    filedialog.asksaveasfilename = lambda **k: proj
    filedialog.askopenfilename = lambda **k: proj

    def load():
        a.save_project()
        a.load_project()
    op("儲存 -> 載入專案", load, "快取為空")

    def delete():
        a.tree.selection_set(str(a.engine.targets[0].id))
        a.delete_selected()
    op("刪除目標", delete, "目標移除")
    report["E4_invalidation"] = e4
    root.destroy()

    # ---- E2 直接匯出 (不分析); Tk 同一行程只保留一個視窗, 所以放在 E4 關閉之後
    root2, a2 = new_app(path)
    t2 = add_target(a2, gt)
    c2 = Counter()
    c2.install(a2.engine)
    out2 = os.path.join(OUT, "P0-3_export_direct.mp4")
    st2 = batch(a2, out2)
    c2.uninstall()
    diffs = []
    for k, p in cache_after_analysis.items():
        q = t2.cache.get(k, {}).get("poly") if k in t2.cache else "missing"
        if isinstance(q, str):
            diffs.append((k, "missing"))
        elif (p is None) != (q is None):
            diffs.append((k, "none-mismatch"))
        elif p is not None:
            d = float(np.abs(p - q).max())
            if d > 0.5:
                diffs.append((k, round(d, 3)))
    max_d = max([float(np.abs(p - t2.cache[k]["poly"]).max()) for k, p in cache_after_analysis.items()
                 if p is not None and k in t2.cache and t2.cache[k]["poly"] is not None] or [0])
    report["E2_direct_export"] = {"error": st2["error"], "wall_s": st2["wall_s"], "track_step_calls": c2.track,
                                  "recover_calls": c2.recover, "poly_max_diff_px": round(max_d, 4),
                                  "frames_diff_gt_0.5px_or_mismatch": diffs[:20], "n_diffs": len(diffs),
                                  "output_compare": frames_psnr(out1, out2)}
    print("E2", report["E2_direct_export"], flush=True)
    root2.destroy()

    report["E3_seek_alignment"] = e3_seek(vids)
    save_json(os.path.join(OUT, "P0-3_report.json"), report)


if __name__ == "__main__":
    main()
