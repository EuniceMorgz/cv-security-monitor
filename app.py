"""
app.py - Desktop application for the Motion-Triggered Security Monitor
CSC 126 Computer Vision - Midterm Project

A Tkinter window that shows the live camera (or a video file) with boxes, IDs and
trails, the motion mask, statistics, an event log, and control buttons.
All computer-vision work is done by cv_engine.py.

Run:   python app.py
Build: pyinstaller --onefile --windowed --name MotionMonitor app.py
"""
import os
import sys
import time
import datetime
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

from cv_engine import MotionMonitor, FRAME_W, FRAME_H

try:
    import winsound            # built into Windows, used for the alert beep
except ImportError:
    winsound = None

SNAPSHOT_COOLDOWN = 3          # seconds between saved alert snapshots
QUIET_FRAMES = 20              # frames without motion before an event is "ended"
FRAME_DELAY_MS = 10            # pause between frames in the GUI loop
MASK_PREVIEW = (320, 240)      # size of the small mask preview


def app_folder():
    """Folder of the program: next to the .exe when built, or next to this file."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


class MotionMonitorApp:
    """The main window and its behavior."""

    def __init__(self, root):
        self.root = root
        root.title("Motion-Triggered Security Monitor")
        root.resizable(False, False)

        self.monitor = MotionMonitor()
        self.cap = None                # the camera or video file
        self.running = False
        self.after_id = None           # id of the next scheduled frame update
        self.source_is_camera = True

        self.last_save = 0             # time of the last saved snapshot
        self.in_event = False          # True while motion is ongoing
        self.quiet_count = 0           # frames in a row without motion
        self.prev_time = time.time()
        self.last_combined = None      # last frame + mask, used for screenshots

        self.captures_dir = os.path.join(app_folder(), "captures")
        self.screens_dir = os.path.join(app_folder(), "screenshots")
        os.makedirs(self.captures_dir, exist_ok=True)
        os.makedirs(self.screens_dir, exist_ok=True)

        self.build_ui()
        self.show_placeholder()
        root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ------------------------------------------------------------------
    # Building the window
    # ------------------------------------------------------------------
    def build_ui(self):
        main = ttk.Frame(self.root, padding=8)
        main.grid()

        # ----- Left column: video, statistics, event log -----
        left = ttk.Frame(main)
        left.grid(row=0, column=0, sticky="n")

        self.video_label = ttk.Label(left)
        self.video_label.pack()

        stats = ttk.Frame(left, padding=(0, 6))
        stats.pack(fill="x")
        self.status_var = tk.StringVar(value="Stopped")
        self.fps_var = tk.StringVar(value="FPS: 0")
        self.view_var = tk.StringVar(value="In view: 0")
        self.total_var = tk.StringVar(value="Total IDs: 0")
        self.status_label = ttk.Label(stats, textvariable=self.status_var,
                                      font=("Segoe UI", 11, "bold"), width=22)
        self.status_label.grid(row=0, column=0, sticky="w")
        ttk.Label(stats, textvariable=self.fps_var, width=10).grid(row=0, column=1)
        ttk.Label(stats, textvariable=self.view_var, width=12).grid(row=0, column=2)
        ttk.Label(stats, textvariable=self.total_var, width=14).grid(row=0, column=3)

        log_frame = ttk.LabelFrame(left, text="Event log", padding=4)
        log_frame.pack(fill="x")
        self.log_list = tk.Listbox(log_frame, height=4, width=95)
        scroll = ttk.Scrollbar(log_frame, command=self.log_list.yview)
        self.log_list.configure(yscrollcommand=scroll.set)
        self.log_list.pack(side="left", fill="x", expand=True)
        scroll.pack(side="right", fill="y")

        # ----- Right column: mask preview and controls -----
        right = ttk.Frame(main)
        right.grid(row=0, column=1, sticky="n", padx=(10, 0))

        ttk.Label(right, text="Motion Mask (what the program sees)",
                  font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.mask_label = ttk.Label(right)
        self.mask_label.pack(pady=(2, 8))

        buttons = ttk.LabelFrame(right, text="Controls", padding=6)
        buttons.pack(fill="x")
        ttk.Button(buttons, text="Start Webcam", command=self.start_webcam
                   ).grid(row=0, column=0, sticky="ew", padx=2, pady=2)
        ttk.Button(buttons, text="Open Video File", command=self.open_video
                   ).grid(row=0, column=1, sticky="ew", padx=2, pady=2)
        ttk.Button(buttons, text="Stop", command=self.stop
                   ).grid(row=1, column=0, sticky="ew", padx=2, pady=2)
        ttk.Button(buttons, text="Reset Background", command=self.reset_background
                   ).grid(row=1, column=1, sticky="ew", padx=2, pady=2)
        ttk.Button(buttons, text="Save Screenshot", command=self.save_screenshot
                   ).grid(row=2, column=0, columnspan=2, sticky="ew", padx=2, pady=2)
        buttons.columnconfigure(0, weight=1)
        buttons.columnconfigure(1, weight=1)

        sliders = ttk.LabelFrame(right, text="Settings", padding=6)
        sliders.pack(fill="x", pady=8)
        ttk.Label(sliders, text="Motion threshold (higher = less sensitive)").pack(anchor="w")
        self.threshold_scale = tk.Scale(sliders, from_=20, to=200, orient="horizontal",
                                        command=self.on_threshold_change)
        self.threshold_scale.set(self.monitor.var_threshold)
        self.threshold_scale.pack(fill="x")
        ttk.Label(sliders, text="Minimum object size (pixels)").pack(anchor="w")
        self.area_scale = tk.Scale(sliders, from_=500, to=10000, resolution=100,
                                   orient="horizontal", command=self.on_area_change)
        self.area_scale.set(self.monitor.min_area)
        self.area_scale.pack(fill="x")

        self.sound_var = tk.BooleanVar(value=True)
        self.save_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(right, text="Beep on alert", variable=self.sound_var).pack(anchor="w")
        ttk.Checkbutton(right, text="Save snapshot on alert", variable=self.save_var).pack(anchor="w")

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------
    def log(self, message):
        """Add a time-stamped line to the event log."""
        stamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_list.insert(tk.END, f"{stamp}  {message}")
        self.log_list.see(tk.END)

    def show_placeholder(self):
        """Show a black picture with instructions before anything is started."""
        blank = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
        cv2.putText(blank, "Press 'Start Webcam' or 'Open Video File'", (60, FRAME_H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        self.show(blank, np.zeros_like(blank))

    def show(self, frame_bgr, mask_bgr):
        """Display the annotated frame and the mask preview in the window."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self.video_img = ImageTk.PhotoImage(Image.fromarray(rgb))   # keep a reference
        self.video_label.configure(image=self.video_img)

        small = cv2.resize(mask_bgr, MASK_PREVIEW)
        self.mask_img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
        self.mask_label.configure(image=self.mask_img)

    def set_status(self, text, color):
        self.status_var.set(text)
        self.status_label.configure(foreground=color)

    # ------------------------------------------------------------------
    # Slider callbacks
    # ------------------------------------------------------------------
    def on_threshold_change(self, value):
        self.monitor.set_threshold(int(float(value)))

    def on_area_change(self, value):
        self.monitor.min_area = int(float(value))

    # ------------------------------------------------------------------
    # Starting and stopping
    # ------------------------------------------------------------------
    def start_webcam(self):
        self.stop()
        # CAP_DSHOW makes webcams open more reliably on Windows
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            messagebox.showerror(
                "Camera error",
                "Cannot open the webcam.\n\n"
                "- Close other apps that use the camera (Zoom, Teams, Messenger)\n"
                "- Check Settings > Privacy & security > Camera\n"
                "- Or use 'Open Video File' instead")
            return
        self.begin(cap, is_camera=True, name="Webcam")

    def open_video(self):
        path = filedialog.askopenfilename(
            title="Choose a video file",
            initialdir=os.path.join(app_folder(), "samples"),
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")])
        if not path:
            return
        self.stop()
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            messagebox.showerror("Video error", f"Cannot open this video:\n{path}")
            return
        self.begin(cap, is_camera=False, name=os.path.basename(path))

    def begin(self, cap, is_camera, name):
        """Start processing frames from the given camera or video."""
        self.cap = cap
        self.source_is_camera = is_camera
        self.monitor.reset()           # learn a fresh background
        self.in_event = False
        self.quiet_count = 0
        self.prev_time = time.time()
        self.running = True
        self.log(f"Started: {name}")
        self.update_frame()

    def stop(self):
        """Stop processing and release the camera or video."""
        was_running = self.running
        self.running = False
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if was_running:
            self.set_status("Stopped", "black")
            self.log("Stopped")

    def reset_background(self):
        """Make the program learn the background again (stay out of view)."""
        self.monitor.reset()
        self.in_event = False
        self.log("Background reset - stay out of view for a moment")

    # ------------------------------------------------------------------
    # The main loop: runs once per frame
    # ------------------------------------------------------------------
    def update_frame(self):
        if not self.running:
            return
        ok, frame = self.cap.read()
        if not ok:
            ended_camera = self.source_is_camera
            self.stop()
            if ended_camera:
                messagebox.showerror("Camera error", "The camera stopped sending pictures.")
            else:
                self.log("Video finished")
                self.set_status("Video finished", "black")
            return

        annotated, mask, info = self.monitor.process(frame)
        self.last_combined = np.hstack((annotated, mask))

        self.handle_alert(annotated, info)
        self.update_stats(info)
        self.show(annotated, mask)
        self.after_id = self.root.after(FRAME_DELAY_MS, self.update_frame)

    def update_stats(self, info):
        now = time.time()
        fps = 1.0 / max(now - self.prev_time, 1e-6)
        self.prev_time = now
        self.fps_var.set(f"FPS: {fps:.0f}")
        self.view_var.set(f"In view: {info['in_view']}")
        self.total_var.set(f"Total IDs: {info['total_ids']}")
        if info["calibrating"]:
            self.set_status("Calibrating...", "#b8860b")
        elif info["alert"]:
            self.set_status("ALERT: MOTION DETECTED", "red")
        else:
            self.set_status("Clear - no motion", "green")

    def handle_alert(self, annotated, info):
        """Log motion events, save snapshots, and play the beep."""
        if info["alert"]:
            self.quiet_count = 0
            if not self.in_event:
                self.in_event = True
                self.log(f"Motion detected ({info['in_view']} object(s) in view)")
            if time.time() - self.last_save > SNAPSHOT_COOLDOWN:
                self.last_save = time.time()
                if self.save_var.get():
                    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(os.path.join(self.captures_dir, f"alert_{stamp}.jpg"), annotated)
                    self.log(f"Snapshot saved: alert_{stamp}.jpg")
                if self.sound_var.get() and winsound is not None:
                    # run in a thread so the beep does not freeze the video
                    threading.Thread(target=winsound.Beep, args=(1000, 200),
                                     daemon=True).start()
        elif self.in_event:
            self.quiet_count += 1
            if self.quiet_count >= QUIET_FRAMES:
                self.in_event = False
                self.log("Motion ended")

    # ------------------------------------------------------------------
    # Screenshot and closing
    # ------------------------------------------------------------------
    def save_screenshot(self):
        if self.last_combined is None:
            messagebox.showinfo("Screenshot", "Start the webcam or a video first.")
            return
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.screens_dir, f"shot_{stamp}.png")
        cv2.imwrite(path, self.last_combined)
        self.log(f"Screenshot saved: shot_{stamp}.png")

    def on_close(self):
        self.stop()
        self.root.destroy()


def main():
    root = tk.Tk()
    MotionMonitorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
