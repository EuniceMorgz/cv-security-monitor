"""
Motion-Triggered Security Monitor with Multi-Object Tracking
CSC 126 Computer Vision - Midterm Project

Techniques used:
  a. Image Operations   - resize, Gaussian blur, morphological opening/dilation
  b. Image Segmentation - MOG2 background subtraction + binary thresholding
  e. Motion & Tracking  - contours, bounding boxes, centroid tracker (IDs + trails)

Controls: press 'q' to quit, 's' to save a screenshot.
"""
import os
import time
import datetime
from collections import OrderedDict

import cv2
import numpy as np

# ---------------- Configuration (tune these during testing) ----------------
SOURCE = 0           # 0 = webcam, or a path like "samples/test.mp4"
FRAME_W, FRAME_H = 640, 480
WARMUP_FRAMES = 30         # frames used to learn the background
MIN_AREA = 3000           # ignore moving blobs smaller than this (noise)
MAX_MISSING = 40           # frames an object may vanish before its ID is dropped
MAX_DISTANCE = 120         # max pixels an object may move between frames
TRAIL_LENGTH = 30          # number of past positions drawn as a trail
SNAPSHOT_COOLDOWN = 3      # seconds between saved alert snapshots


class CentroidTracker:
    """Tracks objects by matching centroids between consecutive frames."""

    def __init__(self, max_missing=MAX_MISSING, max_distance=MAX_DISTANCE):
        self.next_id = 0
        self.objects = OrderedDict()   # id -> (cx, cy)
        self.missing = {}              # id -> consecutive frames not seen
        self.trails = {}               # id -> list of past centroids
        self.max_missing = max_missing
        self.max_distance = max_distance

    def register(self, centroid):
        """Give a new object the next free ID."""
        self.objects[self.next_id] = centroid
        self.missing[self.next_id] = 0
        self.trails[self.next_id] = [centroid]
        self.next_id += 1

    def deregister(self, object_id):
        """Forget an object that has been missing for too long."""
        del self.objects[object_id]
        del self.missing[object_id]
        del self.trails[object_id]

    def update(self, centroids):
        """Match this frame's centroids to existing objects."""
        # Case 1: nothing detected -> every known object is missing one more frame
        if len(centroids) == 0:
            for oid in list(self.objects.keys()):
                self.missing[oid] += 1
                if self.missing[oid] > self.max_missing:
                    self.deregister(oid)
            return

        # Case 2: nothing tracked yet -> register everything
        if len(self.objects) == 0:
            for c in centroids:
                self.register(c)
            return

        # Case 3: match old objects to new centroids by smallest distance
        ids = list(self.objects.keys())
        old = np.array([self.objects[i] for i in ids])
        new = np.array(centroids)
        dist = np.linalg.norm(old[:, None, :] - new[None, :, :], axis=2)

        rows = dist.min(axis=1).argsort()      # closest old objects first
        cols = dist.argmin(axis=1)[rows]
        used_rows, used_cols = set(), set()

        for r, c in zip(rows, cols):
            if r in used_rows or c in used_cols or dist[r, c] > self.max_distance:
                continue
            oid = ids[r]
            self.objects[oid] = centroids[c]
            self.missing[oid] = 0
            self.trails[oid].append(centroids[c])
            self.trails[oid] = self.trails[oid][-TRAIL_LENGTH:]
            used_rows.add(r)
            used_cols.add(c)

        # Old objects with no match: count as missing
        for r in set(range(len(ids))) - used_rows:
            oid = ids[r]
            self.missing[oid] += 1
            if self.missing[oid] > self.max_missing:
                self.deregister(oid)

        # New centroids with no match: brand new objects
        for c in set(range(len(centroids))) - used_cols:
            self.register(centroids[c])


def open_source(source):
    """Open a webcam (DirectShow on Windows) or a video file."""
    if isinstance(source, int):
        return cv2.VideoCapture(source, cv2.CAP_DSHOW)
    return cv2.VideoCapture(source)


def preprocess(frame):
    """Topic a: resize and blur the frame to reduce noise."""
    frame = cv2.resize(frame, (FRAME_W, FRAME_H))
    blurred = cv2.GaussianBlur(frame, (9, 9), 0)
    return frame, blurred


def segment_motion(bg_subtractor, blurred, kernel):
    """Topic b: background subtraction, thresholding, and morphology."""
    mask = bg_subtractor.apply(blurred)
    _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)  # removes shadows (gray)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)       # remove specks
    mask = cv2.dilate(mask, kernel, iterations=4)               # fill holes
    return mask


def find_objects(mask):
    """Topic e: find contours and return bounding boxes of large moving blobs."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= MIN_AREA]


def merge_boxes(boxes, gap=60):
    """Merge boxes that overlap or are close together, so one person = one box."""
    boxes = [list(b) for b in boxes]
    merged = True
    while merged:
        merged = False
        result = []
        while boxes:
            x, y, w, h = boxes.pop()
            i = 0
            while i < len(boxes):
                bx, by, bw, bh = boxes[i]
                # do the two boxes touch (with a small gap allowed)?
                if (x - gap < bx + bw and bx - gap < x + w and
                        y - gap < by + bh and by - gap < y + h):
                    nx, ny = min(x, bx), min(y, by)
                    w = max(x + w, bx + bw) - nx
                    h = max(y + h, by + bh) - ny
                    x, y = nx, ny
                    boxes.pop(i)
                    merged = True
                else:
                    i += 1
            result.append([x, y, w, h])
        boxes = result
    return [tuple(b) for b in boxes]


def draw_results(frame, boxes, tracker):
    """Draw bounding boxes, ID labels, centroids, and movement trails."""
    # Map each object's center point to its ID, so each box can show its ID
    id_by_centroid = {c: oid for oid, c in tracker.objects.items()
                      if tracker.missing[oid] == 0}

    for (x, y, w, h) in boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        center = (x + w // 2, y + h // 2)
        oid = id_by_centroid.get(center)
        if oid is not None:
            label = f"ID {oid}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            top = max(y - th - 10, 0)
            # black background so the label is always readable
            cv2.rectangle(frame, (x, top), (x + tw + 8, top + th + 8), (0, 0, 0), -1)
            cv2.putText(frame, label, (x + 4, top + th + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    for oid, (cx, cy) in tracker.objects.items():
        if tracker.missing[oid] > 0:          # only draw objects seen this frame
            continue
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
        pts = tracker.trails[oid]
        for i in range(1, len(pts)):
            cv2.line(frame, pts[i - 1], pts[i], (255, 200, 0), 2)


def main(source=SOURCE):
    cap = open_source(source)
    if not cap.isOpened():
        print("Error: cannot open video source:", source)
        return

    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=100)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    tracker = CentroidTracker()
    os.makedirs("captures", exist_ok=True)
    os.makedirs("screenshots", exist_ok=True)

    frame_count, last_save, prev_time = 0, 0, time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break                              # end of video or camera error
        frame_count += 1

        frame, blurred = preprocess(frame)
        mask = segment_motion(bg_subtractor, blurred, kernel)
        boxes = []

        if frame_count <= WARMUP_FRAMES:
            cv2.putText(frame, "Calibrating background...", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        else:
            boxes = merge_boxes(find_objects(mask))
            centroids = [(x + w // 2, y + h // 2) for (x, y, w, h) in boxes]
            tracker.update(centroids)
            draw_results(frame, boxes, tracker)

            if boxes:                          # alert + snapshot
                cv2.putText(frame, "ALERT: MOTION DETECTED", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                if time.time() - last_save > SNAPSHOT_COOLDOWN:
                    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(f"captures/alert_{stamp}.jpg", frame)
                    last_save = time.time()

        # Status bar: FPS, objects in view, total unique objects tracked
        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        active = sum(1 for m in tracker.missing.values() if m == 0)
        cv2.putText(frame, f"FPS: {fps:.0f}  In view: {active}  Total IDs: {tracker.next_id}",
                    (10, FRAME_H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Side-by-side display: processed frame | motion mask
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.putText(mask_bgr, "Motion Mask", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        combined = np.hstack((frame, mask_bgr))
        cv2.imshow("Motion & Object Tracking", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"screenshots/shot_{stamp}.png", combined)
            print("Saved screenshot", stamp)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()