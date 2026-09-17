"""影片追蹤模糊工具 (Windows / macOS)

操作:
  開啟影片 -> 暫停在要處理的畫面 -> 「新增目標」框選 (矩形拖曳, 或四點依序點窗戶四角)
  播放時會自動追蹤並模糊; 相機轉開再轉回來會自動找回
  「從此幀取消」: 從目前這幀起停止追蹤並取消模糊 (之前的保留)
  「刪除目標」: 整段都不模糊
  匯出影片前會從頭追蹤一次, 所以不必每一幀都預覽過
快捷鍵: 空白鍵 播放/暫停, ←/→ 前後一幀, Shift+←/→ 前後 1 秒, N 新增矩形, Q 新增四點,
        S 從此幀取消, B 模糊開關, Delete 刪除目標, Esc 取消框選
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

import tracker as T

BLUR_MODES = [("高斯模糊", "gaussian"), ("馬賽克", "pixelate"), ("黑色遮擋", "solid")]
IS_MAC = sys.platform == "darwin"
WORK_WIDTHS = [("960 (快)", 960), ("1440", 1440), ("1920", 1920), ("原始解析度 (慢)", 0)]
ENCODERS = [
    ("H.264 軟體 x264 (畫質最佳)", "libx264"),
    ("H.265 軟體 x265 (檔案小, 慢)", "libx265"),
    ("H.264 Apple 硬體", "h264_videotoolbox"),
    ("H.265 Apple 硬體", "hevc_videotoolbox"),
    ("H.264 NVIDIA 硬體", "h264_nvenc"),
    ("H.265 NVIDIA 硬體", "hevc_nvenc"),
]


def available_encoders():
    """回傳 UI 可選的 [(名稱, ffmpeg 編碼器)], 第一個為預設. Mac 預設用 Apple 硬體編碼"""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return [("OpenCV mp4v (未安裝 ffmpeg)", "opencv")]
    try:
        kw = {"creationflags": 0x08000000} if sys.platform == "win32" else {}
        out = subprocess.run([ffmpeg, "-hide_banner", "-encoders"], stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL, timeout=15, **kw).stdout.decode("utf-8", "replace")
        names = set(line.split()[1] for line in out.splitlines() if len(line.split()) > 1)
    except (OSError, subprocess.SubprocessError):
        names = {"libx264"}
    encs = [e for e in ENCODERS if e[1] in names] or [ENCODERS[0]]
    if IS_MAC:
        encs.sort(key=lambda e: e[1] != "h264_videotoolbox")
    return encs


def video_args(encoder, w, h, fps, src_kbps):
    """ffmpeg 視訊編碼參數. 軟體編碼用固定畫質 (CRF); 硬體編碼用位元率 (參考原片, 至少依解析度估計)"""
    kbps = int(max(src_kbps * 1.2 if src_kbps else 0, w * h * fps * 0.1 / 1000.0, 2000))
    hevc_tag = ["-tag:v", "hvc1"] if "265" in encoder or "hevc" in encoder else []
    if encoder == "libx264":
        args = ["-c:v", "libx264", "-crf", "18", "-preset", "medium"]
    elif encoder == "libx265":
        args = ["-c:v", "libx265", "-crf", "20", "-preset", "medium"]
    elif encoder.endswith("_videotoolbox"):
        args = ["-c:v", encoder, "-b:v", "%dk" % kbps, "-maxrate", "%dk" % int(kbps * 1.5),
                "-bufsize", "%dk" % (kbps * 2), "-allow_sw", "1"]
    elif encoder.endswith("_nvenc"):
        args = ["-c:v", encoder, "-preset", "p5", "-rc", "vbr", "-cq", "19" if "h264" in encoder else "21",
                "-b:v", "0"]
    else:
        raise ValueError("未知的編碼器: %s" % encoder)
    return args + ["-pix_fmt", "yuv420p"] + hevc_tag


class App(object):
    def __init__(self, root):
        self.root = root
        root.title("影片追蹤模糊")
        root.geometry("1400x860")
        root.minsize(1000, 600)

        self.engine = T.Engine()
        self.cap = None
        self.path = None
        self.n_frames = 0
        self.fps = 30.0
        self.cur_idx = -1
        self.cap_pos = -1        # cap 下一次 read() 會讀到的幀號
        self.cur_frame = None
        self.cur_gray = None
        self.prev_gray = None
        self.scale = 1.0
        self.results = []
        self.playing = False
        self.busy = False

        self.draw_mode = None    # None / "rect" / "quad"
        self.redraw_target = None
        self.draw_pts = []
        self.drag_start = None
        self.disp = (1.0, 0, 0)  # 顯示縮放, x 偏移, y 偏移
        self._photo = None
        self._slider_guard = False
        self._pending_seek = None

        self._build_ui()
        self._bind_keys()
        self._set_status("請開啟影片")

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        root = self.root
        bar = ttk.Frame(root, padding=(6, 4))
        bar.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(bar, text="開啟影片", command=self.open_video).pack(side=tk.LEFT)
        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)
        ttk.Button(bar, text="◀", width=3, command=lambda: self.step(-1)).pack(side=tk.LEFT)
        self.play_btn = ttk.Button(bar, text="▶ 播放", width=8, command=self.toggle_play)
        self.play_btn.pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="▶", width=3, command=lambda: self.step(1)).pack(side=tk.LEFT)
        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)
        ttk.Button(bar, text="分析追蹤(全片)", command=lambda: self.run_batch(None)).pack(side=tk.LEFT)
        ttk.Button(bar, text="匯出影片", command=self.export_video).pack(side=tk.LEFT, padx=4)
        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)
        ttk.Button(bar, text="儲存專案", command=self.save_project).pack(side=tk.LEFT)
        ttk.Button(bar, text="載入專案", command=self.load_project).pack(side=tk.LEFT, padx=4)

        body = ttk.Frame(root)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # ---- 右側面板
        side = ttk.Frame(body, padding=6, width=330)
        side.pack(side=tk.RIGHT, fill=tk.Y)
        side.pack_propagate(False)

        ttk.Label(side, text="追蹤目標", font=("", 12, "bold")).pack(anchor=tk.W)
        cols = ("name", "status", "blur")
        self.tree = ttk.Treeview(side, columns=cols, show="headings", height=9, selectmode="browse")
        for c, txt, w in (("name", "名稱", 80), ("status", "狀態", 150), ("blur", "模糊", 50)):
            self.tree.heading(c, text=txt)
            self.tree.column(c, width=w, anchor=tk.W if c != "blur" else tk.CENTER)
        self.tree.pack(fill=tk.X, pady=(2, 6))
        self.tree.bind("<<TreeviewSelect>>", lambda e: (self.render(), self.draw_timeline()))
        self.tree.bind("<Double-1>", lambda e: self.toggle_blur())

        g = ttk.Frame(side)
        g.pack(fill=tk.X)
        btns = [
            ("＋ 新增目標 (拖曳矩形) [N]", lambda: self.start_draw("rect")),
            ("＋ 新增目標 (點四角) [Q]", lambda: self.start_draw("quad")),
            ("在此幀重新框選 (修正位置)", self.redraw_selected),
            ("⏹ 從此幀取消追蹤+模糊 [S]", self.stop_selected),
            ("↺ 恢復追蹤 (移除前一個取消點)", self.resume_selected),
            ("模糊 開/關 (整段) [B]", self.toggle_blur),
            ("🗑 刪除目標 [Del]", self.delete_selected),
        ]
        for txt, cmd in btns:
            ttk.Button(g, text=txt, command=cmd).pack(fill=tk.X, pady=1)

        ttk.Separator(side).pack(fill=tk.X, pady=8)
        ttk.Label(side, text="模糊設定", font=("", 12, "bold")).pack(anchor=tk.W)
        f = ttk.Frame(side)
        f.pack(fill=tk.X)
        f.columnconfigure(1, weight=1)

        ttk.Label(f, text="方式").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.mode_var = tk.StringVar(value=BLUR_MODES[0][0])
        cb = ttk.Combobox(f, textvariable=self.mode_var, values=[m[0] for m in BLUR_MODES], state="readonly")
        cb.grid(row=0, column=1, sticky=tk.EW)
        cb.bind("<<ComboboxSelected>>", lambda e: self._apply_settings())

        self.strength_var = tk.IntVar(value=self.engine.strength)
        self._slider_row(f, 1, "強度", self.strength_var, 1, 100)
        self.pad_var = tk.IntVar(value=int(self.engine.padding * 100))
        self._slider_row(f, 2, "外擴 %", self.pad_var, 0, 50)

        ttk.Label(f, text="遺失後保留(幀)").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.hold_var = tk.IntVar(value=self.engine.hold_frames)
        sp = ttk.Spinbox(f, from_=0, to=600, textvariable=self.hold_var, width=6, command=self._apply_settings)
        sp.grid(row=3, column=1, sticky=tk.W)
        sp.bind("<Return>", lambda e: self._apply_settings())
        sp.bind("<FocusOut>", lambda e: self._apply_settings())

        self.conf_var = tk.IntVar(value=int(self.engine.min_conf * 100))
        self._slider_row(f, 4, "最低可信度 %", self.conf_var, 0, 90)

        ttk.Label(f, text="追蹤解析度").grid(row=5, column=0, sticky=tk.W, pady=2)
        self.work_var = tk.StringVar(value=self._work_label(self.engine.work_width))
        cb = ttk.Combobox(f, textvariable=self.work_var, values=[w[0] for w in WORK_WIDTHS], state="readonly")
        cb.grid(row=5, column=1, sticky=tk.EW)
        cb.bind("<<ComboboxSelected>>", lambda e: self.change_work_width())

        ttk.Label(f, text="輸出編碼").grid(row=6, column=0, sticky=tk.W, pady=2)
        self.encoders = available_encoders()
        self.enc_var = tk.StringVar(value=self.encoders[0][0])
        ttk.Combobox(f, textvariable=self.enc_var, values=[e[0] for e in self.encoders], state="readonly").grid(
            row=6, column=1, sticky=tk.EW)

        self.show_box_var = tk.BooleanVar(value=True)
        self.preview_blur_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(f, text="顯示追蹤框", variable=self.show_box_var, command=self.render).grid(
            row=7, column=0, sticky=tk.W)
        ttk.Checkbutton(f, text="預覽模糊", variable=self.preview_blur_var, command=self.render).grid(
            row=7, column=1, sticky=tk.W)

        ttk.Separator(side).pack(fill=tk.X, pady=8)
        tip = ("說明:\n"
               "• 暫停在目標清楚的畫面再框選\n"
               "• 窗戶建議用「點四角」, 轉向時會跟著透視變形\n"
               "• 綠框=追蹤中 (數字為可信度), 橘框=暫時遺失仍模糊, 紅虛線=遺失已停止模糊\n"
               "• 跳到後面沒追蹤過的幀不會模糊, 按「分析追蹤」或匯出時會自動追蹤\n"
               "• 可信度低於門檻視為遺失, 避免模糊到另一扇相似的窗戶\n"
               "• 追歪了: 暫停, 選目標, 按「在此幀重新框選」")
        ttk.Label(side, text=tip, wraplength=310, justify=tk.LEFT, foreground="#555").pack(anchor=tk.W)

        # ---- 中央畫面與時間軸
        center = ttk.Frame(body)
        center.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(center, bg="#202020", highlightthickness=0, cursor="arrow")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda e: self.render())
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Motion>", self.on_motion)
        self.canvas.bind("<Button-2>" if IS_MAC else "<Button-3>", self.on_right_click)

        tl = ttk.Frame(center, padding=(6, 2))
        tl.pack(fill=tk.X)
        self.slider = ttk.Scale(tl, from_=0, to=1, orient=tk.HORIZONTAL, command=self.on_slider)
        self.slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.time_lbl = ttk.Label(tl, text="0 / 0", width=28, anchor=tk.E)
        self.time_lbl.pack(side=tk.RIGHT)
        self.timeline = tk.Canvas(center, height=14, bg="#e8e8e8", highlightthickness=0)
        self.timeline.pack(fill=tk.X, padx=6)
        self.timeline.bind("<Configure>", lambda e: self.draw_timeline())

        self.status = ttk.Label(root, anchor=tk.W, padding=(6, 3), relief=tk.SUNKEN)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

    def _slider_row(self, parent, row, label, var, lo, hi):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, pady=2)
        fr = ttk.Frame(parent)
        fr.grid(row=row, column=1, sticky=tk.EW)
        val = ttk.Label(fr, width=4, text=str(var.get()))
        val.pack(side=tk.RIGHT)

        def changed(v):
            var.set(int(float(v)))
            val.config(text=str(var.get()))
            self._apply_settings()
        s = ttk.Scale(fr, from_=lo, to=hi, orient=tk.HORIZONTAL, value=var.get())
        s.config(command=changed)
        s.pack(side=tk.LEFT, fill=tk.X, expand=True)

    def _bind_keys(self):
        r = self.root
        keys = {
            "<space>": lambda e: self.toggle_play(),
            "<Left>": lambda e: self.step(-1),
            "<Right>": lambda e: self.step(1),
            "<Shift-Left>": lambda e: self.step(-int(round(self.fps))),
            "<Shift-Right>": lambda e: self.step(int(round(self.fps))),
            "<n>": lambda e: self.start_draw("rect"),
            "<q>": lambda e: self.start_draw("quad"),
            "<s>": lambda e: self.stop_selected(),
            "<b>": lambda e: self.toggle_blur(),
            "<Delete>": lambda e: self.delete_selected(),
            "<BackSpace>": lambda e: self.delete_selected(),
            "<Escape>": lambda e: self.cancel_draw(),
        }
        for k, fn in keys.items():
            r.bind_all(k, self._key_guard(fn))

    def _key_guard(self, fn):
        def h(e):
            w = self.root.focus_get()
            if isinstance(w, (tk.Entry, ttk.Entry, ttk.Spinbox, ttk.Combobox)) and e.keysym != "Escape":
                return None
            if self.busy:
                return "break"
            fn(e)
            return "break"
        return h

    def _set_status(self, txt):
        self.status.config(text=txt)

    def _apply_settings(self):
        self.engine.blur_mode = dict(BLUR_MODES)[self.mode_var.get()]
        self.engine.strength = self.strength_var.get()
        self.engine.padding = self.pad_var.get() / 100.0
        min_conf = self.conf_var.get() / 100.0
        if min_conf != self.engine.min_conf:
            # 可信度門檻會改變追蹤結果, 之前算好的要重算
            self.engine.min_conf = min_conf
            for t in self.engine.targets:
                t.cache.clear()
        try:
            self.engine.hold_frames = max(0, int(self.hold_var.get()))
        except (tk.TclError, ValueError):
            pass
        if self.cur_frame is not None:
            self.results = self.engine.process(self.cur_idx, self.cur_gray, self.prev_gray, self.scale)
            self.render()

    @staticmethod
    def _work_label(width):
        return next((lbl for lbl, w in WORK_WIDTHS if w == width), WORK_WIDTHS[0][0])

    def change_work_width(self):
        width = dict(WORK_WIDTHS)[self.work_var.get()]
        if width == self.engine.work_width:
            return
        self.engine.work_width = width
        if self.cap is None:
            return
        # 樣板是用舊解析度建立的, 需重建; 追蹤結果也要重算
        self._set_status("重建追蹤樣板中...")
        self.root.update_idletasks()
        self._rebuild_refs()
        idx, self.cur_idx, self.cap_pos = self.cur_idx, -1, -1
        self.goto(idx)
        self._set_status("追蹤解析度: %s (已清除舊的追蹤結果)" % self.work_var.get())

    def _rebuild_refs(self):
        cap = cv2.VideoCapture(self.path)
        for t in self.engine.targets:
            t.cache.clear()
            for k, v in t.keyframes.items():
                if v is T.STOP:
                    continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, k)
                ok, fr = cap.read()
                if ok:
                    g, s = self.engine.to_gray(fr)
                    t.refs[k] = T.build_ref(g, v * s)
        cap.release()

    # ------------------------------------------------------------ video io
    def open_video(self, path=None):
        path = path or filedialog.askopenfilename(
            title="開啟影片", filetypes=[("影片", "*.mp4 *.mov *.avi *.mkv *.m4v *.wmv *.MP4 *.MOV"), ("全部", "*.*")])
        if not path:
            return False
        cap = cv2.VideoCapture(path)
        ok, frame = cap.read()
        if not ok:
            messagebox.showerror("錯誤", "無法讀取影片:\n%s" % path)
            return False
        if self.cap is not None:
            self.cap.release()
        self.pause()
        self.cap, self.path = cap, path
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if not (1 < self.fps < 240):
            self.fps = 30.0
        self.n_frames = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        self.engine.targets = []
        self.cap_pos = 1
        self.cur_idx = -1
        self._set_frame(0, frame, None)
        self.slider.config(to=max(1, self.n_frames - 1))
        self.refresh_tree()
        self.root.title("影片追蹤模糊 - %s" % os.path.basename(path))
        h, w = frame.shape[:2]
        self._set_status("%s  %dx%d  %.2f fps  %d 幀" % (os.path.basename(path), w, h, self.fps, self.n_frames))
        return True

    def _read(self, idx):
        if idx != self.cap_pos:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, f = self.cap.read()
        self.cap_pos = idx + 1 if ok else -1
        return f if ok else None

    def goto(self, idx):
        if self.cap is None:
            return False
        idx = int(max(0, min(idx, self.n_frames - 1)))
        if idx == self.cur_idx:
            return True
        prev_gray = None
        if idx == self.cur_idx + 1:
            prev_gray = self.cur_gray
        elif idx > 0:
            pf = self._read(idx - 1)
            if pf is not None:
                prev_gray = self.engine.to_gray(pf)[0]
        f = self._read(idx)
        if f is None:
            # 影格數估計不準時, 以實際可讀到的為準
            if idx > 0 and idx >= self.n_frames - 5:
                self.n_frames = idx
                self.slider.config(to=max(1, self.n_frames - 1))
            return False
        self._set_frame(idx, f, prev_gray)
        return True

    def _set_frame(self, idx, frame, prev_gray):
        self.cur_idx = idx
        self.cur_frame = frame
        self.cur_gray, self.scale = self.engine.to_gray(frame)
        self.prev_gray = prev_gray
        self.results = self.engine.process(idx, self.cur_gray, prev_gray, self.scale)
        self.render()
        self._update_time()
        self.refresh_tree(statuses_only=True)

    def _update_time(self):
        self._slider_guard = True
        self.slider.set(self.cur_idx)
        self._slider_guard = False
        t = self.cur_idx / self.fps
        tt = self.n_frames / self.fps
        self.time_lbl.config(text="%02d:%05.2f / %02d:%05.2f   #%d" % (t // 60, t % 60, tt // 60, tt % 60, self.cur_idx))

    def on_slider(self, v):
        if self._slider_guard or self.cap is None:
            return
        self.pause()
        first = self._pending_seek is None
        self._pending_seek = int(float(v))
        if first:
            self.root.after(40, self._do_seek)

    def _do_seek(self):
        idx, self._pending_seek = self._pending_seek, None
        if idx is not None:
            self.goto(idx)

    def step(self, d):
        if self.cap is None:
            return
        self.pause()
        self.cancel_draw()
        self.goto(self.cur_idx + d)

    def toggle_play(self):
        if self.playing:
            self.pause()
        elif self.cap is not None:
            self.cancel_draw()
            if self.cur_idx >= self.n_frames - 1:
                self.goto(0)
            self.playing = True
            self.play_btn.config(text="⏸ 暫停")
            self._play_tick()

    def pause(self):
        self.playing = False
        self.play_btn.config(text="▶ 播放")

    def _play_tick(self):
        if not self.playing:
            return
        t0 = time.time()
        if not self.goto(self.cur_idx + 1) or self.cur_idx >= self.n_frames - 1:
            self.pause()
            return
        delay = int(1000.0 / self.fps - (time.time() - t0) * 1000)
        self.root.after(max(1, delay), self._play_tick)

    # -------------------------------------------------------------- render
    def selected_target(self):
        sel = self.tree.selection()
        if not sel:
            return None
        tid = int(sel[0])
        for t in self.engine.targets:
            if t.id == tid:
                return t
        return None

    def render(self):
        c = self.canvas
        c.delete("all")
        if self.cur_frame is None:
            return
        cw, ch = max(c.winfo_width(), 10), max(c.winfo_height(), 10)
        fh, fw = self.cur_frame.shape[:2]
        s = min(cw / float(fw), ch / float(fh))
        dw, dh = max(1, int(fw * s)), max(1, int(fh * s))
        ox, oy = (cw - dw) // 2, (ch - dh) // 2
        self.disp = (s, ox, oy)

        img = self.cur_frame.copy()
        if self.preview_blur_var.get():
            self.engine.render(img, self.results)
        img = cv2.resize(img, (dw, dh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        self._photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        c.create_image(ox, oy, anchor=tk.NW, image=self._photo)

        if self.show_box_var.get():
            sel = self.selected_target()
            for t, poly, status in self.results:
                label = t.name
                dash = () if t.blur_enabled else (4, 3)
                if poly is None:
                    e = t.cache.get(self.cur_idx)
                    if e is None or e["lost"] == 0:
                        continue
                    # 遺失且已停止模糊: 在最後位置畫紅色虛線提醒
                    poly, color, dash = e["anchor"], "#ff4d4f", (6, 4)
                    label = "%s 遺失-未模糊" % t.name
                elif status.startswith("追蹤中"):
                    color = "#34d058"
                    label = "%s%s" % (t.name, status[3:])
                else:
                    color = "#ff9f1a"
                if not t.blur_enabled:
                    color = "#9aa0a6"
                pts = [(x * s + ox, y * s + oy) for x, y in poly]
                c.create_polygon(*sum(pts, ()), outline=color, fill="", width=3 if t is sel else 1.5, dash=dash)
                x0, y0 = min(p[0] for p in pts), min(p[1] for p in pts)
                c.create_text(x0 + 2, y0 - 2, anchor=tk.SW, text=label, fill=color, font=("", 11, "bold"))
        self._draw_temp()

    def _draw_temp(self, mouse=None):
        c = self.canvas
        c.delete("temp")
        if not self.draw_mode:
            return
        s, ox, oy = self.disp
        pts = [(x * s + ox, y * s + oy) for x, y in self.draw_pts]
        if mouse is not None and self.draw_mode == "quad" and pts:
            pts = pts + [mouse]
        if len(pts) >= 2:
            c.create_line(*sum(pts, ()), fill="#00e5ff", width=2, tags="temp")
        for x, y in pts[:len(self.draw_pts)]:
            c.create_oval(x - 4, y - 4, x + 4, y + 4, outline="#00e5ff", width=2, tags="temp")
        what = "重新框選 " + self.redraw_target.name if self.redraw_target else "新增目標"
        hint = ("%s: 拖曳滑鼠框出範圍  (Esc 取消)" if self.draw_mode == "rect"
                else "%s: 依序點擊四個角 (%d/4)  右鍵退回一點, Esc 取消" % ("%s", len(self.draw_pts)))
        c.create_text(12, 12, anchor=tk.NW, text=hint % what, fill="#00e5ff", font=("", 13, "bold"), tags="temp")

    def draw_timeline(self):
        c = self.timeline
        c.delete("all")
        t = self.selected_target()
        w = max(c.winfo_width(), 10)
        if t is None or self.n_frames <= 1:
            return
        n = self.n_frames
        ks = sorted(t.keyframes)
        for i, k in enumerate(ks):
            if t.keyframes[k] is T.STOP:
                continue
            end = next((kk for kk in ks[i + 1:] if t.keyframes[kk] is T.STOP), n)
            c.create_rectangle(k / float(n) * w, 3, end / float(n) * w, 11, fill="#9be9a8", width=0)
        for k in ks:
            x = k / float(n) * w
            c.create_rectangle(x - 1, 0, x + 1, 14, width=0,
                               fill="#d73a49" if t.keyframes[k] is T.STOP else "#1a7f37")

    # ------------------------------------------------------------- targets
    def refresh_tree(self, statuses_only=False):
        st = dict((t.id, s) for t, _, s in self.results)
        if not statuses_only:
            sel = self.tree.selection()
            self.tree.delete(*self.tree.get_children())
            for t in self.engine.targets:
                self.tree.insert("", tk.END, iid=str(t.id), values=(t.name, "", ""))
            if sel and self.tree.exists(sel[0]):
                self.tree.selection_set(sel[0])
        for t in self.engine.targets:
            if self.tree.exists(str(t.id)):
                self.tree.item(str(t.id), values=(t.name, st.get(t.id, ""), "開" if t.blur_enabled else "關"))
        if not statuses_only:
            self.draw_timeline()

    def _need_target(self):
        t = self.selected_target()
        if t is None:
            if len(self.engine.targets) == 1:
                t = self.engine.targets[0]
                self.tree.selection_set(str(t.id))
            else:
                self._set_status("請先在右側清單選擇一個目標")
        return t

    def _after_edit(self):
        self.results = self.engine.process(self.cur_idx, self.cur_gray, self.prev_gray, self.scale)
        self.refresh_tree()
        self.render()

    def start_draw(self, mode, target=None):
        if self.cur_frame is None:
            return
        self.pause()
        self.draw_mode = mode
        self.redraw_target = target
        self.draw_pts = []
        self.canvas.config(cursor="crosshair")
        self._draw_temp()

    def cancel_draw(self):
        if self.draw_mode:
            self.draw_mode = None
            self.draw_pts = []
            self.canvas.config(cursor="arrow")
            self.canvas.delete("temp")

    def redraw_selected(self):
        t = self._need_target()
        if t is not None:
            kf = t.active_keyframe(self.cur_idx)
            mode = "quad"
            if kf is not None and isinstance(t.keyframes[kf], np.ndarray):
                p = t.keyframes[kf]
                if len(p) == 4 and p[0][1] == p[1][1] and p[1][0] == p[2][0]:
                    mode = "rect"
            self.start_draw(mode, t)

    def _to_frame(self, x, y):
        s, ox, oy = self.disp
        fh, fw = self.cur_frame.shape[:2]
        return (min(max((x - ox) / s, 0), fw - 1), min(max((y - oy) / s, 0), fh - 1))

    def on_press(self, e):
        if not self.draw_mode:
            return
        if self.draw_mode == "rect":
            self.drag_start = (e.x, e.y)
        else:
            self.draw_pts.append(self._to_frame(e.x, e.y))
            if len(self.draw_pts) == 4:
                self._commit(np.float32(self._order_quad(self.draw_pts)))
            else:
                self._draw_temp((e.x, e.y))

    def on_drag(self, e):
        if self.draw_mode == "rect" and self.drag_start:
            self.canvas.delete("temp_rect")
            self.canvas.create_rectangle(self.drag_start[0], self.drag_start[1], e.x, e.y,
                                         outline="#00e5ff", width=2, tags=("temp", "temp_rect"))

    def on_release(self, e):
        if self.draw_mode == "rect" and self.drag_start:
            (x0, y0), (x1, y1) = self._to_frame(*self.drag_start), self._to_frame(e.x, e.y)
            self.drag_start = None
            x0, x1 = sorted((x0, x1))
            y0, y1 = sorted((y0, y1))
            if (x1 - x0) * self.disp[0] < 6 or (y1 - y0) * self.disp[0] < 6:
                self._draw_temp()
                return
            self._commit(np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]))

    def on_motion(self, e):
        if self.draw_mode == "quad" and self.draw_pts:
            self._draw_temp((e.x, e.y))

    def on_right_click(self, e):
        if self.draw_mode == "quad" and self.draw_pts:
            self.draw_pts.pop()
            self._draw_temp((e.x, e.y))

    @staticmethod
    def _order_quad(pts):
        """把使用者任意順序點的四點排成不自交的順時針順序"""
        p = np.float32(pts)
        c = p.mean(axis=0)
        ang = np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0])
        return p[np.argsort(ang)]

    def _commit(self, poly):
        t = self.redraw_target
        if t is None or t not in self.engine.targets:
            t = T.Target()
            self.engine.targets.append(t)
        t.set_keyframe(self.cur_idx, poly, self.cur_gray, self.scale)
        self.cancel_draw()
        self.refresh_tree()
        self.tree.selection_set(str(t.id))
        self._after_edit()
        self._set_status("%s: 已於第 %d 幀設定, 按空白鍵播放開始追蹤" % (t.name, self.cur_idx))

    def stop_selected(self):
        t = self._need_target()
        if t is None:
            return
        t.stop_at(self.cur_idx)
        self._after_edit()
        self._set_status("%s: 從第 %d 幀起取消追蹤與模糊 (之前的片段保留)" % (t.name, self.cur_idx))

    def resume_selected(self):
        t = self._need_target()
        if t is None:
            return
        stops = [k for k, v in t.keyframes.items() if v is T.STOP and k <= self.cur_idx]
        if not stops:
            self._set_status("%s: 此幀之前沒有取消點" % t.name)
            return
        k = max(stops)
        del t.keyframes[k]
        t.invalidate(k)
        self._after_edit()
        self._set_status("%s: 已移除第 %d 幀的取消點, 從前面播放或按「分析追蹤」即可接續" % (t.name, k))

    def toggle_blur(self):
        t = self._need_target()
        if t is not None:
            t.blur_enabled = not t.blur_enabled
            self.refresh_tree(statuses_only=True)
            self.render()

    def delete_selected(self):
        t = self._need_target()
        if t is not None:
            self.engine.targets.remove(t)
            self._after_edit()

    # ---------------------------------------------------- batch / export
    def export_video(self):
        if self.cap is None:
            return
        if not self.engine.targets:
            if not messagebox.askyesno("匯出", "沒有任何追蹤目標, 仍要匯出嗎?"):
                return
        base = os.path.splitext(os.path.basename(self.path))[0]
        out = filedialog.asksaveasfilename(title="匯出影片", defaultextension=".mp4",
                                           initialfile=base + "_blur.mp4", filetypes=[("MP4", "*.mp4")])
        if out:
            self.run_batch(out)

    def run_batch(self, out_path):
        """從頭追蹤整部影片; out_path 不為 None 時同時輸出模糊後影片"""
        if self.cap is None or self.busy:
            return
        self.pause()
        self.cancel_draw()
        self.busy = True
        dlg = tk.Toplevel(self.root)
        dlg.title("匯出中" if out_path else "分析追蹤中")
        dlg.transient(self.root)
        dlg.resizable(False, False)
        lbl = ttk.Label(dlg, text="準備中...", padding=10, width=50)
        lbl.pack()
        pb = ttk.Progressbar(dlg, maximum=self.n_frames, length=380)
        pb.pack(padx=10)
        state = {"i": 0, "done": False, "cancel": False, "error": None, "msg": ""}
        ttk.Button(dlg, text="取消", command=lambda: state.update(cancel=True)).pack(pady=10)
        dlg.protocol("WM_DELETE_WINDOW", lambda: state.update(cancel=True))
        dlg.update_idletasks()
        try:
            dlg.grab_set()
        except tk.TclError:
            pass

        state["encoder"] = dict(self.encoders).get(self.enc_var.get(), "libx264")
        th = threading.Thread(target=self._batch_worker, args=(self.path, out_path, state))
        th.daemon = True
        th.start()
        t0 = time.time()

        def poll():
            i = state["i"]
            total = state.get("total", self.n_frames)
            pb.config(maximum=max(1, total), value=i)
            el = time.time() - t0
            fps = i / el if el > 0 else 0
            lbl.config(text=state["msg"] or "%d / %d 幀   %.1f fps" % (i, total, fps))
            if not state["done"]:
                self.root.after(100, poll)
                return
            try:
                dlg.grab_release()
            except tk.TclError:
                pass
            dlg.destroy()
            self.busy = False
            idx = self.cur_idx
            self.cur_idx = -1
            self.cap_pos = -1
            self.goto(idx)
            self.refresh_tree()
            if state["error"]:
                messagebox.showerror("錯誤", state["error"])
            elif state["cancel"]:
                self._set_status("已取消")
            elif out_path:
                self._set_status("匯出完成: %s" % out_path)
                messagebox.showinfo("完成", "匯出完成:\n%s%s" % (out_path, state.get("note", "")))
            else:
                self._set_status("分析完成, 可任意拖曳時間軸檢查追蹤結果")
        poll()

    def analysis_range(self):
        """只需分析的幀範圍 [start, end): 從最早的框選到最後一個取消點 (沒取消則到片尾)"""
        start, end = None, 0
        for t in self.engine.targets:
            polys = [k for k, v in t.keyframes.items() if v is not T.STOP]
            if not polys:
                continue
            start = min(polys) if start is None else min(start, min(polys))
            last = max(t.keyframes)
            end = max(end, last if t.keyframes[last] is T.STOP else self.n_frames)
        return (0, 0) if start is None else (start, end)

    def _batch_worker(self, src, out_path, state):
        sink = None
        try:
            cap = cv2.VideoCapture(src)
            start, end = 0, None
            if not out_path:  # 只分析: 跳過不需要追蹤的片段
                start, end = self.analysis_range()
                state["total"] = end - start
                if start > 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            src_kbps = cap.get(getattr(cv2, "CAP_PROP_BITRATE", 47)) or 0
            prev = None
            idx = start
            while not state["cancel"] and (end is None or idx < end):
                ok, frame = cap.read()
                if not ok:
                    break
                gray, s = self.engine.to_gray(frame)
                res = self.engine.process(idx, gray, prev, s)
                if out_path:
                    if sink is None:
                        h, w = frame.shape[:2]
                        sink = VideoSink(out_path, src, w, h, self.fps, state.get("encoder", "libx264"), src_kbps)
                    sink.write(self.engine.render(frame, res))
                prev = gray
                idx += 1
                state["i"] = idx - start
            cap.release()
            if out_path and idx > 0 and not state["cancel"]:
                self.n_frames = idx
            if sink is not None:
                state["msg"] = "寫入檔案中..."
                sink.close(discard=state["cancel"])
                state["note"] = sink.note
                sink = None
        except Exception as ex:  # noqa
            state["error"] = str(ex)
            if sink is not None:
                sink.close(discard=True)
        finally:
            state["done"] = True

    # ------------------------------------------------------------- project
    def save_project(self):
        if self.cap is None:
            return
        path = filedialog.asksaveasfilename(title="儲存專案", defaultextension=".json",
                                            initialfile=os.path.splitext(os.path.basename(self.path))[0] + "_blur.json",
                                            filetypes=[("專案", "*.json")])
        if not path:
            return
        data = {
            "video": self.path,
            "settings": {"mode": self.engine.blur_mode, "strength": self.engine.strength,
                         "padding": self.engine.padding, "hold": self.engine.hold_frames,
                         "min_conf": self.engine.min_conf, "work_width": self.engine.work_width},
            "targets": [{
                "name": t.name, "blur": t.blur_enabled,
                "keyframes": [[k, "STOP" if v is T.STOP else v.tolist()] for k, v in sorted(t.keyframes.items())],
            } for t in self.engine.targets],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        self._set_status("已儲存專案: %s" % path)

    def load_project(self):
        path = filedialog.askopenfilename(title="載入專案", filetypes=[("專案", "*.json")])
        if not path:
            return
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        video = data["video"]
        if not os.path.exists(video):
            video = os.path.join(os.path.dirname(path), os.path.basename(video))
        if not os.path.exists(video):
            messagebox.showinfo("找不到影片", "請重新選擇專案對應的影片:\n%s" % data["video"])
            video = None
        st = data.get("settings", {})
        # 追蹤解析度要在讀影片、建樣板之前設定
        self.engine.work_width = int(st.get("work_width", T.WORK_WIDTH))
        self.work_var.set(self._work_label(self.engine.work_width))
        if not self.open_video(video):
            return
        self.mode_var.set(dict((v, k) for k, v in BLUR_MODES).get(st.get("mode"), BLUR_MODES[0][0]))
        self.strength_var.set(st.get("strength", 50))
        self.pad_var.set(int(round(st.get("padding", 0.05) * 100)))
        self.hold_var.set(st.get("hold", 5))
        self.conf_var.set(int(round(st.get("min_conf", 0.3) * 100)))
        cap = cv2.VideoCapture(video)
        for td in data["targets"]:
            t = T.Target(td["name"])
            t.blur_enabled = td.get("blur", True)
            for k, v in td["keyframes"]:
                if v == "STOP":
                    t.stop_at(int(k))
                    continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(k))
                ok, fr = cap.read()
                if ok:
                    g, s = self.engine.to_gray(fr)
                    t.set_keyframe(int(k), np.float32(v), g, s)
            self.engine.targets.append(t)
        cap.release()
        self._apply_settings()
        self.refresh_tree()
        self._set_status("已載入專案, 建議先按「分析追蹤」")


class VideoSink(object):
    """輸出影片, 只壓縮一次.
    有 ffmpeg: 未壓縮的畫面直接用管線送進 ffmpeg, 壓成 H.264 並合併原影片音訊.
    沒有 ffmpeg: 用 OpenCV 直接寫 mp4 (無聲音, 畫質較低)."""

    def __init__(self, out_path, src, w, h, fps, encoder="libx264", src_kbps=0):
        self.out_path = out_path
        self.note = ""
        self.proc = None
        self.writer = None
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg and encoder != "opencv":
            self.log = tempfile.TemporaryFile()
            cmd = ([ffmpeg, "-y", "-loglevel", "error",
                    "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "%dx%d" % (w, h), "-r", "%.6f" % fps, "-i", "-",
                    "-i", src, "-map", "0:v:0", "-map", "1:a?"]
                   + video_args(encoder, w, h, fps, src_kbps)
                   + ["-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", out_path])
            kw = {"creationflags": 0x08000000} if sys.platform == "win32" else {}
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                         stderr=self.log, **kw)
        else:
            self.writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            if not self.writer.isOpened():
                raise RuntimeError("無法建立輸出檔: %s" % out_path)
            self.note = "\n\n(未找到 ffmpeg: 輸出影片不含聲音, 畫質較低. 建議安裝 ffmpeg)"

    def _ffmpeg_error(self):
        self.log.seek(0)
        return "ffmpeg 錯誤:\n" + self.log.read().decode("utf-8", "replace")[-1500:]

    def write(self, frame):
        if self.writer is not None:
            self.writer.write(frame)
            return
        try:
            self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        except (BrokenPipeError, OSError):
            self.proc.wait()
            raise RuntimeError(self._ffmpeg_error())

    def close(self, discard=False):
        if self.writer is not None:
            self.writer.release()
        elif self.proc is not None:
            try:
                self.proc.stdin.close()
            except OSError:
                pass
            # 不用 kill: chocolatey 等安裝的 ffmpeg 是啟動器, kill 只會停掉外殼. 關閉輸入後 ffmpeg 會自行結束
            code = self.proc.wait()
            if code != 0 and not discard:
                raise RuntimeError(self._ffmpeg_error())
        if discard:
            for _ in range(20):
                try:
                    if os.path.exists(self.out_path):
                        os.remove(self.out_path)
                    break
                except OSError:
                    time.sleep(0.25)


def main():
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:  # noqa
            pass
    root = tk.Tk()
    app = App(root)
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        root.after(100, lambda: app.open_video(sys.argv[1]))
    root.mainloop()


if __name__ == "__main__":
    main()
