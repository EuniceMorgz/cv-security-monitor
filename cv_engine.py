"""
cv_engine.py - Computer-vision engine for the Motion-Triggered Security Monitor
CSC 126 Computer Vision - Midterm Project

This file has NO window code. It only turns one camera frame into:
  1. an annotated frame (boxes, IDs, trails, alert text)
  2. a motion mask
  3. a small dictionary of statistics

Techniques used:
  a. Image Operations   - resize, Gaussian blur, morphological opening/dilation
  b. Image Segmentation - MOG2 background subtraction + binary thresholding
  e. Motion & Tracking  - contours, bounding boxes, centroid tracker (IDs + trails)
"""
from collections import OrderedDict

import cv2
import numpy as np

# ---------------- Settings (tuned during testing) ----------------
FRAME_W, FRAME_H = 640, 480   # every frame is resized to this size
WARMUP_FRAMES = 30            # frames used to learn the empty background
MAX_MISSING = 40              # frames an object may vanish before its ID is dropped
MAX_DISTANCE = 120            # max pixels an object may move between frames
TRAIL_LENGTH = 30             # number of past positions drawn as a trail
MERGE_GAP = 60                # boxes closer than this (pixels) are merged into one


class CentroidTracker:
    """Tracks objects by matching box centers (centroids) between frames."""

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
        """Match this frame's centroids to the objects we already know."""
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

        # New centroids with no match: brand-new objects
        for c in set(range(len(centroids))) - used_cols:
            self.register(centroids[c])


class MotionMonitor:
    """The full pipeline: frame in -> annotated frame, mask, and statistics out."""

    def __init__(self, min_area=3000, var_threshold=100):
        self.min_area = min_area               # ignore moving blobs smaller than this
        self.var_threshold = var_threshold     # higher = less sensitive to change
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.reset()

    def reset(self):
        """Forget the background and all tracked objects, and start learning again."""
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=self.var_threshold)
        self.tracker = CentroidTracker()
        self.frame_count = 0

    def set_threshold(self, value):
        """Change motion sensitivity while the program is running."""
        self.var_threshold = value
        self.bg_subtractor.setVarThreshold(value)

    # ------------------------------------------------------------------
    # Topic a: Image Operations
    # ------------------------------------------------------------------
    @staticmethod
    def preprocess(frame):
        """Resize the frame and blur a copy of it to reduce camera noise."""
        frame = cv2.resize(frame, (FRAME_W, FRAME_H))
        blurred = cv2.GaussianBlur(frame, (9, 9), 0)
        return frame, blurred

    # ------------------------------------------------------------------
    # Topic b: Image Segmentation
    # ------------------------------------------------------------------
    def segment_motion(self, blurred):
        """Background subtraction -> threshold -> morphology -> clean motion mask."""
        mask = self.bg_subtractor.apply(blurred)
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)  # drops gray shadows
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)  # remove specks
        mask = cv2.dilate(mask, self.kernel, iterations=4)          # fill holes
        return mask

    # ------------------------------------------------------------------
    # Topic e: Motion and Object Tracking
    # ------------------------------------------------------------------
    def find_objects(self, mask):
        """Find contours and return bounding boxes of the large moving blobs."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [cv2.boundingRect(c) for c in contours
                if cv2.contourArea(c) >= self.min_area]

    @staticmethod
    def merge_boxes(boxes, gap=MERGE_GAP):
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

    def draw_results(self, frame, boxes):
        """Draw bounding boxes, ID labels, centers, and movement trails."""
        tracker = self.tracker
        # Map each object's center point to its ID so each box can show its ID
        id_by_centroid = {c: oid for oid, c in tracker.objects.items()
                          if tracker.missing[oid] == 0}

        for (x, y, w, h) in boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            oid = id_by_centroid.get((x + w // 2, y + h // 2))
            if oid is not None:
                label = f"ID {oid}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                top = max(y - th - 10, 0)
                # black background so the label is always readable
                cv2.rectangle(frame, (x, top), (x + tw + 8, top + th + 8), (0, 0, 0), -1)
                cv2.putText(frame, label, (x + 4, top + th + 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        for oid, (cx, cy) in tracker.objects.items():
            if tracker.missing[oid] > 0:       # only draw objects seen this frame
                continue
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            pts = tracker.trails[oid]
            for i in range(1, len(pts)):
                cv2.line(frame, pts[i - 1], pts[i], (255, 200, 0), 2)

    # ------------------------------------------------------------------
    # The whole pipeline for one frame
    # ------------------------------------------------------------------
    def process(self, frame):
        """Run the full pipeline. Returns (annotated_frame, mask_bgr, info)."""
        frame, blurred = self.preprocess(frame)
        mask = self.segment_motion(blurred)
        self.frame_count += 1

        boxes = []
        calibrating = self.frame_count <= WARMUP_FRAMES
        if calibrating:
            cv2.putText(frame, "Calibrating background...", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        else:
            boxes = self.merge_boxes(self.find_objects(mask))
            centroids = [(x + w // 2, y + h // 2) for (x, y, w, h) in boxes]
            self.tracker.update(centroids)
            self.draw_results(frame, boxes)
            if boxes:
                cv2.putText(frame, "ALERT: MOTION DETECTED", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        in_view = sum(1 for m in self.tracker.missing.values() if m == 0)
        total_ids = self.tracker.next_id
        cv2.putText(frame, f"In view: {in_view}  Total IDs: {total_ids}",
                    (10, FRAME_H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        info = {
            "calibrating": calibrating,
            "alert": len(boxes) > 0,
            "in_view": in_view,
            "total_ids": total_ids,
        }
        return frame, mask_bgr, info
