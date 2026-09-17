"""P0-4 輸出影片: 幀數 / 解析度 / FPS / 音訊 / 音畫同步
來源影片在同一時間點放「全白閃光幀」與「1 kHz 嗶聲」, 匯出後比對兩者時間差是否改變.
"""
import os
import subprocess
import time
import tkinter as tk
import wave

import cv2
import numpy as np

from common import FFMPEG, NOWIN, OUT, T, env_info, ffprobe_json, read_frames, save_json, write_video

SR = 48000


def make_wav(path, duration, beep_times):
    a = np.zeros(int(duration * SR), np.float32)
    n = int(0.04 * SR)
    tone = 0.8 * np.sin(2 * np.pi * 1000 * np.arange(n) / SR)
    for t in beep_times:
        i = int(round(t * SR))
        a[i:i + n] = tone[:max(0, min(n, len(a) - i))]
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((a * 32767).astype(np.int16).tobytes())


def content(w, h, n, fps, events, seed=1, t0=0.0):
    """移動紋理 + 在事件時間放全白幀. 回傳 generator"""
    rng = np.random.default_rng(seed)
    base = cv2.GaussianBlur((rng.random((h // 4, w // 2, 3)) * 255).astype(np.uint8), (0, 0), 2)
    base = cv2.resize(base, (w * 2, h), interpolation=cv2.INTER_LINEAR)
    flash = set(int(round((t - t0) * fps)) for t in events if t0 <= t)
    for i in range(n):
        if i in flash:
            yield np.full((h, w, 3), 255, np.uint8)
        else:
            x = int(i * 4) % w
            yield (base[:, x:x + w] * 0.8).astype(np.uint8)


def build_sources():
    srcs = {}
    dur = 60
    ev = [2.5 + 5 * k for k in range(12)]

    p = os.path.join(OUT, "S1_1080p30_cfr.mp4")
    if not os.path.exists(p):
        wav = os.path.join(OUT, "S1.wav")
        make_wav(wav, dur, ev)
        write_video(p, content(1920, 1080, dur * 30, 30, ev), 1920, 1080, "30", audio_wav=wav,
                    vcodec=("-c:v", "libx264", "-crf", "18", "-preset", "fast"))
    srcs["S1 1080p 30fps CFR"] = (p, ev)

    p = os.path.join(OUT, "S2_1080p2997_cfr.mp4")
    if not os.path.exists(p):
        wav = os.path.join(OUT, "S2.wav")
        fps = 30000 / 1001.0
        ev2 = [round(round(t * fps) / fps, 6) for t in ev]  # 對齊到幀時間
        make_wav(wav, dur, ev2)
        write_video(p, content(1920, 1080, int(dur * fps), fps, ev2), 1920, 1080, "30000/1001", audio_wav=wav,
                    vcodec=("-c:v", "libx264", "-crf", "18", "-preset", "fast"))
    srcs["S2 1080p 29.97fps CFR"] = (p, ev)

    p = os.path.join(OUT, "S3_1080p_vfr.mp4")
    if not os.path.exists(p):
        segs, evs, t_abs = [], [], 0.0
        for j, fps in enumerate([30, 60, 24]):
            n = 20 * fps
            local = [2.5 + 5 * k for k in range(4)]
            evs += [t_abs + t for t in local]
            sp = os.path.join(OUT, "S3_seg%d.mp4" % j)
            write_video(sp, content(1920, 1080, n, fps, local, seed=j), 1920, 1080, str(fps),
                        vcodec=("-c:v", "libx264", "-crf", "18", "-preset", "fast"), extra=("-video_track_timescale", "90000"))
            segs.append(sp)
            t_abs += n / float(fps)
        lst = os.path.join(OUT, "S3_list.txt")
        with open(lst, "w") as f:
            f.writelines("file '%s'\n" % os.path.basename(s) for s in segs)
        vonly = os.path.join(OUT, "S3_video_only.mp4")
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", vonly], cwd=OUT, **NOWIN)
        wav = os.path.join(OUT, "S3.wav")
        make_wav(wav, t_abs, evs)
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", vonly, "-i", wav, "-map", "0:v", "-map", "1:a",
                        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", p], **NOWIN)
        np.save(p + ".events.npy", np.array(evs))
    srcs["S3 1080p VFR 30/60/24"] = (p, list(np.load(p + ".events.npy")))

    p = os.path.join(OUT, "S4_4k60_cfr.mp4")
    ev4 = [1.5 + 3 * k for k in range(3)]
    if not os.path.exists(p):
        wav = os.path.join(OUT, "S4.wav")
        make_wav(wav, 10, ev4)
        write_video(p, content(3840, 2160, 600, 60, ev4), 3840, 2160, "60", audio_wav=wav,
                    vcodec=("-c:v", "libx264", "-crf", "20", "-preset", "veryfast"))
    srcs["S4 4K 60fps CFR"] = (p, ev4)

    p = os.path.join(OUT, "S5_portrait_rotate90.mp4")
    ev5 = [1.5 + 3 * k for k in range(3)]
    if not os.path.exists(p):
        wav = os.path.join(OUT, "S5.wav")
        make_wav(wav, 10, ev5)
        write_video(p, content(1920, 1080, 300, 30, ev5), 1920, 1080, "30", audio_wav=wav,
                    vcodec=("-c:v", "libx264", "-crf", "18", "-preset", "fast"), extra=("-metadata:s:v:0", "rotate=90"))
    srcs["S5 直式 (rotate=90 metadata)"] = (p, ev5)

    p = os.path.join(OUT, "S6_hevc10_hdrtag.mp4")
    ev6 = [1.5 + 3 * k for k in range(3)]
    if not os.path.exists(p):
        wav = os.path.join(OUT, "S6.wav")
        make_wav(wav, 10, ev6)
        write_video(p, content(1920, 1080, 300, 30, ev6), 1920, 1080, "30", audio_wav=wav,
                    vcodec=("-c:v", "libx265", "-crf", "18", "-preset", "fast", "-x265-params", "log-level=error"),
                    extra=("-pix_fmt", "yuv420p10le", "-color_primaries", "bt2020", "-color_trc", "smpte2084",
                           "-colorspace", "bt2020nc", "-tag:v", "hvc1"))
    srcs["S6 HEVC 10-bit HDR 標記"] = (p, ev6)
    return srcs


# ---------------------------------------------------------------- 量測
def probe(path):
    j = ffprobe_json(path, "-count_frames", "-show_streams", "-show_format")
    v = next((s for s in j.get("streams", []) if s["codec_type"] == "video"), {})
    a = next((s for s in j.get("streams", []) if s["codec_type"] == "audio"), None)
    rot = 0
    for sd in v.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = int(sd["rotation"])
    rot = rot or int((v.get("tags") or {}).get("rotate", 0))
    w, h = v.get("width"), v.get("height")
    disp = (h, w) if abs(rot) % 180 == 90 else (w, h)
    return {
        "codec": v.get("codec_name"), "pix_fmt": v.get("pix_fmt"), "size": [w, h], "rotation": rot, "display_size": list(disp),
        "r_frame_rate": v.get("r_frame_rate"), "avg_frame_rate": v.get("avg_frame_rate"),
        "frames": int(v.get("nb_read_frames", 0) or 0), "v_duration": float(v.get("duration", 0) or 0),
        "v_start": float(v.get("start_time", 0) or 0),
        "color_transfer": v.get("color_transfer"), "color_primaries": v.get("color_primaries"),
        "audio": None if a is None else {"codec": a.get("codec_name"), "duration": float(a.get("duration", 0) or 0),
                                         "start": float(a.get("start_time", 0) or 0), "sample_rate": a.get("sample_rate")},
        "size_mb": round(float(j.get("format", {}).get("size", 0)) / 1e6, 2),
    }


def video_pts(path):
    out = subprocess.run([os.path.join(os.path.dirname(FFMPEG), "ffprobe"), "-v", "error", "-select_streams", "v:0",
                          "-show_entries", "frame=best_effort_timestamp_time", "-of", "csv=p=0", path],
                         stdout=subprocess.PIPE, **NOWIN).stdout.decode()
    return [float(x.strip().strip(",")) for x in out.split() if x.strip().strip(",") not in ("", "N/A")]


def flash_times(path):
    pts = video_pts(path)
    idx = [i for i, f in enumerate(read_frames(path)) if f[::8, ::8].mean() > 235]
    return [pts[i] for i in idx if i < len(pts)], len(pts)


def beep_times(path, a_start):
    raw = subprocess.run([FFMPEG, "-v", "error", "-i", path, "-map", "0:a:0", "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"],
                         stdout=subprocess.PIPE, **NOWIN).stdout
    x = np.abs(np.frombuffer(raw, np.int16).astype(np.float32))
    if len(x) == 0:
        return []
    env = np.convolve(x, np.ones(96) / 96, mode="same")
    on = env > 0.25 * env.max()
    t, last = [], -1.0
    for i in np.flatnonzero(on[1:] & ~on[:-1]) + 1:
        ts = i / float(SR)
        if ts - last > 0.5:
            t.append(ts + a_start)
            last = ts
    return t


def av_offsets(path, pr):
    fl, _ = flash_times(path)
    bp = beep_times(path, pr["audio"]["start"]) if pr["audio"] else []
    pairs = []
    for f in fl:
        if bp:
            b = min(bp, key=lambda x: abs(x - f))
            if abs(b - f) < 1.5:
                pairs.append((round(f, 4), round(b, 4), round(f - b, 4)))
    return pairs


def export(a, src, out, encoder, cancel_after=None):
    a.open_video(src)
    st = {"i": 0, "done": False, "cancel": False, "error": None, "msg": "", "encoder": encoder}
    if cancel_after:
        orig = a.engine.process

        def proc(idx, *x):
            if idx >= cancel_after:
                st["cancel"] = True
            return orig(idx, *x)
        a.engine.process = proc
    t0 = time.time()
    a._batch_worker(src, out, st)
    if cancel_after:
        a.engine.process = orig
    st["wall_s"] = round(time.time() - t0, 1)
    st["app_fps_prop"] = round(a.fps, 4)
    return st


def main():
    import app as A
    srcs = build_sources()
    root = tk.Tk()
    a = A.App(root)
    root.update()
    encs = [e[1] for e in a.encoders]
    report = {"env": env_info(), "available_encoders": encs, "results": []}
    plan = [("S1 1080p 30fps CFR", e) for e in encs] + [(k, "libx264") for k in srcs if not k.startswith("S1")]
    if "h264_nvenc" in encs:
        plan.append(("S4 4K 60fps CFR", "h264_nvenc"))
    src_cache = {}
    for name, enc in plan:
        src, ev = srcs[name]
        if name not in src_cache:
            sp = probe(src)
            src_cache[name] = {"probe": sp, "av": av_offsets(src, sp)}
        sp, sav = src_cache[name]["probe"], src_cache[name]["av"]
        out = os.path.join(OUT, "P0-4_%s_%s.mp4" % (name.split()[0], enc))
        st = export(a, src, out, enc)
        r = {"source": name, "encoder": enc, "error": st["error"], "wall_s": st["wall_s"], "export_fps": round(st["i"] / max(st["wall_s"], 1e-6), 1),
             "opencv_CAP_PROP_FPS": st["app_fps_prop"], "src": sp}
        if not st["error"] and os.path.exists(out):
            op = probe(out)
            oav = av_offsets(out, op)
            fdur = 1.0 / eval(sp["r_frame_rate"]) if sp["r_frame_rate"] and sp["r_frame_rate"] != "0/0" else 1 / 30.0
            src_off = [p[2] for p in sav]
            out_off = [p[2] for p in oav]
            drift = [round(o - s, 4) for s, o in zip(src_off, out_off)]
            r.update({
                "out": op,
                "check_display_size": op["display_size"] == sp["display_size"],
                "check_r_frame_rate": op["r_frame_rate"] == sp["r_frame_rate"],
                "check_frames": op["frames"] == sp["frames"], "frames_diff": op["frames"] - sp["frames"],
                "check_audio": (op["audio"] is not None) == (sp["audio"] is not None),
                "va_duration_diff_s": None if not op["audio"] else round((op["v_start"] + op["v_duration"]) - (op["audio"]["start"] + op["audio"]["duration"]), 4),
                "src_events": len(sav), "out_events": len(oav),
                "src_av_offsets": src_off, "out_av_offsets": out_off, "av_drift_vs_src": drift,
                "max_abs_drift_s": max([abs(d) for d in drift] or [None]) if drift else None,
                "sync_limit_s": round(fdur + 0.023, 4),
            })
            r["check_sync"] = r["max_abs_drift_s"] is not None and r["max_abs_drift_s"] <= r["sync_limit_s"] and len(oav) == len(sav)
            r["check_va_duration"] = r["va_duration_diff_s"] is not None and abs(r["va_duration_diff_s"]) <= r["sync_limit_s"]
        report["results"].append(r)
        print(name, enc, {k: r.get(k) for k in ("error", "check_display_size", "check_r_frame_rate", "check_frames", "frames_diff",
                                                 "check_audio", "check_sync", "max_abs_drift_s", "va_duration_diff_s", "export_fps")}, flush=True)
    # 取消匯出不殘留
    out = os.path.join(OUT, "P0-4_cancel.mp4")
    if os.path.exists(out):
        os.remove(out)
    st = export(a, srcs["S1 1080p 30fps CFR"][0], out, encs[0], cancel_after=60)
    report["cancel"] = {"error": st["error"], "file_exists_after_cancel": os.path.exists(out)}
    print("cancel", report["cancel"], flush=True)
    root.destroy()
    save_json(os.path.join(OUT, "P0-4_report.json"), report)


if __name__ == "__main__":
    main()
