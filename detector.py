import threading
import queue
import time
import cv2
import numpy as np
from pathlib import Path

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class DetectorThread(threading.Thread):

    BOTTLE_CLASS_ID = 39   # COCO class 39 = bottle
    CROP_SIZE = (224, 224)

    def __init__(self, source, crop_queue: queue.Queue):
        super().__init__(daemon=True)
        self.source = source
        self.crop_queue = crop_queue

        self.frame = None
        self.raw_frame = None
        self.count = 0              # unique bottles only — never double-counted
        self.fps = 0.0
        self.done = False
        self.total_frames = 0
        self.processed_frames = 0
        self.running = True

        # Density estimate for pile images
        self.density_estimate = 0
        self.count_method = "yolo"  # "yolo" or "density"

        # ByteTrack: track IDs we have already sent to VLM
        # Only used in video mode — prevents same bottle being classified
        # multiple times across frames
        self._seen_track_ids: set = set()
        self._seen_lock = threading.Lock()

        self.source_type = self._detect_source_type(source)

        if YOLO_AVAILABLE:
            # yolov8m = medium model — much better detection in cluttered/pile
            # scenes than yolov8n (nano). Auto-downloads ~50MB on first run.
            self.model = YOLO("yolov8m.pt")
        else:
            self.model = None

    def _detect_source_type(self, source) -> str:
        if source is None:
            return "mock"
        p = Path(str(source))
        if p.suffix.lower() in SUPPORTED_IMAGE_EXTS:
            return "image"
        if p.is_dir():
            return "image_folder"
        return "video"

    def run(self):
        if self.source_type == "image":
            self._process_image(self.source)
        elif self.source_type == "image_folder":
            self._process_folder(self.source)
        elif self.source_type == "video":
            self._process_video(self.source)
        else:
            self._run_dummy()
        self.done = True

    # ── Single image ──────────────────────────────────────────────────────────
    def _process_image(self, path):
        frame = cv2.imread(str(path))
        if frame is None:
            print(f"[Detector] Could not read image: {path}")
            return

        self.total_frames = 1
        self.raw_frame = frame.copy()

        # Use tiled detection for large images (pile scenes)
        # Any image larger than ~700x700 pixels gets tiled
        h, w = frame.shape[:2]
        if h * w > 500_000:
            self._detect_tiled(frame, frame_num=0)
        else:
            self._detect_image_mode(frame, frame_num=0)

        self.processed_frames = 1

        # Supplement YOLO count with density estimate for pile images
        density_est = self._estimate_density(frame)
        if density_est > self.count * 2:
            self.density_estimate = density_est
            self.count_method = "density"
        else:
            self.density_estimate = self.count
            self.count_method = "yolo"

    # ── Folder of images ──────────────────────────────────────────────────────
    def _process_folder(self, folder):
        paths = sorted([
            p for p in Path(folder).iterdir()
            if p.suffix.lower() in SUPPORTED_IMAGE_EXTS
        ])
        self.total_frames = len(paths)
        for i, path in enumerate(paths):
            if not self.running:
                break
            frame = cv2.imread(str(path))
            if frame is None:
                continue
            h, w = frame.shape[:2]
            if h * w > 500_000:
                self._detect_tiled(frame, frame_num=i)
            else:
                self._detect_image_mode(frame, frame_num=i)
            self.processed_frames = i + 1
            time.sleep(0.05)

    # ── Video — uses ByteTrack to avoid double-counting ───────────────────────
    def _process_video(self, path):
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            print(f"[Detector] Could not open video: {path}")
            self._run_dummy()
            return

        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_num = 0
        t_prev = time.perf_counter()

        while self.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break   # video ended — do NOT loop

            self.raw_frame = frame.copy()
            self._detect_video_frame(frame, frame_num)
            self.processed_frames = frame_num + 1

            t_now = time.perf_counter()
            self.fps = round(1.0 / max(t_now - t_prev, 1e-6), 1)
            t_prev = t_now
            frame_num += 1

        cap.release()

    # ── Detection for single images (no tracker needed) ───────────────────────
    def _detect_image_mode(self, frame, frame_num: int):
        """
        Standard detection on a single frame.
        Each bounding box = one unique object — no deduplication needed.
        """
        if self.model is None:
            self.frame = frame
            return

        results = self.model(
            frame,
            classes=[self.BOTTLE_CLASS_ID],
            verbose=False,
            conf=0.15,          # low threshold to catch occluded/partial bottles
            iou=0.30,           # low IOU = NMS keeps more overlapping boxes
            imgsz=1280,         # larger input = better small object detection
            max_det=300,
            agnostic_nms=True,
        )

        self.frame = results[0].plot()

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if x2 <= x1 or y2 <= y1:
                continue
            crop = frame[y1:y2, x1:x2]
            crop_resized = cv2.resize(crop, self.CROP_SIZE)
            try:
                self.crop_queue.put_nowait((self.count, crop_resized, frame_num))
            except queue.Full:
                pass
            self.count += 1

    # ── Tiled detection for large/cluttered pile images ───────────────────────
    def _detect_tiled(self, frame, frame_num: int, tile_size=640, overlap=100):
        """
        Splits image into overlapping tiles and runs YOLO on each tile.
        Dramatically improves detection count on dense pile images where
        individual bottles are small relative to the full image.

        tile_size: pixels per tile (matches YOLO's native 640 input)
        overlap:   pixels of overlap between adjacent tiles to avoid
                   missing bottles that sit on tile boundaries
        """
        if self.model is None:
            self.frame = frame
            return

        h, w = frame.shape[:2]
        step = tile_size - overlap
        all_boxes = []

        for y in range(0, h, step):
            for x in range(0, w, step):
                x2 = min(x + tile_size, w)
                y2 = min(y + tile_size, h)
                tile = frame[y:y2, x:x2]

                results = self.model(
                    tile,
                    classes=[self.BOTTLE_CLASS_ID],
                    verbose=False,
                    conf=0.15,
                    iou=0.30,
                    agnostic_nms=True,
                )

                for box in results[0].boxes:
                    bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                    # Translate tile-local coords back to full image coords
                    all_boxes.append([bx1 + x, by1 + y, bx2 + x, by2 + y])

        # De-duplicate boxes from adjacent overlapping tiles using NMS
        if all_boxes:
            try:
                import torch
                from torchvision.ops import nms as tv_nms
                boxes_t = torch.tensor(all_boxes, dtype=torch.float32)
                scores = torch.ones(len(all_boxes))
                keep = tv_nms(boxes_t, scores, iou_threshold=0.40)
                kept_boxes = boxes_t[keep].int().tolist()
            except ImportError:
                # torchvision not installed — use all boxes without NMS dedup
                kept_boxes = all_boxes

            # Draw results on frame
            annotated = frame.copy()
            for bx1, by1, bx2, by2 in kept_boxes:
                cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
            cv2.putText(
                annotated,
                f"Tiled detection: {len(kept_boxes)} bottles",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 0), 2
            )
            self.frame = annotated

            # Send each unique crop to VLM queue
            for bx1, by1, bx2, by2 in kept_boxes:
                crop = frame[by1:by2, bx1:bx2]
                if crop.size == 0:
                    continue
                crop_resized = cv2.resize(crop, self.CROP_SIZE)
                try:
                    self.crop_queue.put_nowait((self.count, crop_resized, frame_num))
                except queue.Full:
                    pass
                self.count += 1
        else:
            self.frame = frame

    # ── Video detection WITH ByteTrack (no double-counting) ───────────────────
    def _detect_video_frame(self, frame, frame_num: int):
        """
        Uses YOLO's built-in ByteTrack tracker.
        Each physical bottle gets a persistent track_id across frames.
        We only send a crop to the VLM the FIRST time we see each track_id,
        preventing the same bottle from being classified multiple times.
        """
        if self.model is None:
            self.frame = frame
            return

        # persist=True keeps tracker state alive between calls
        results = self.model.track(
            frame,
            classes=[self.BOTTLE_CLASS_ID],
            verbose=False,
            conf=0.20,
            iou=0.35,
            persist=True,
            tracker="bytetrack.yaml",
        )

        self.frame = results[0].plot()

        if results[0].boxes.id is None:
            return   # no tracked objects this frame

        track_ids = results[0].boxes.id.int().tolist()
        boxes = results[0].boxes.xyxy

        for track_id, box in zip(track_ids, boxes):
            # Skip if we have already classified this bottle in a previous frame
            with self._seen_lock:
                if track_id in self._seen_track_ids:
                    continue
                self._seen_track_ids.add(track_id)

            x1, y1, x2, y2 = map(int, box)
            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame[y1:y2, x1:x2]
            crop_resized = cv2.resize(crop, self.CROP_SIZE)

            try:
                self.crop_queue.put_nowait((self.count, crop_resized, frame_num))
            except queue.Full:
                pass

            self.count += 1   # only increments for new unique track IDs

    # ── Density estimator for pile images ─────────────────────────────────────
    def _estimate_density(self, frame) -> int:
        """
        Fallback count estimator for dense pile images where YOLO undercounts.
        Uses HSV color segmentation to estimate how many bottle-sized objects
        are present based on the fraction of the image covered by plastic.
        Not precise — provides an order-of-magnitude estimate shown alongside
        the YOLO count in the dashboard.
        """
        h, w = frame.shape[:2]
        image_area = h * w

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Clear/white plastic regions
        plastic_mask = cv2.inRange(hsv, (0, 0, 150), (180, 60, 255))
        # Colored labels and caps
        colored_mask = cv2.inRange(hsv, (0, 60, 60), (180, 255, 255))
        combined = cv2.bitwise_or(plastic_mask, colored_mask)

        kernel = np.ones((5, 5), np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

        plastic_pixels = cv2.countNonZero(combined)
        plastic_ratio = plastic_pixels / image_area

        # Assume avg bottle occupies ~1.5% of image area in a pile scene
        avg_bottle_area_ratio = 0.015
        estimated = int(plastic_ratio / avg_bottle_area_ratio)

        return max(estimated, self.count)

    # ── Dummy fallback ────────────────────────────────────────────────────────
    def _run_dummy(self):
        for i in range(10):
            if not self.running:
                break
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "No source — mock mode", (80, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (180, 180, 180), 2)
            self.frame = blank
            try:
                self.crop_queue.put_nowait((i, None, i))
            except queue.Full:
                pass
            self.count = i + 1
            self.total_frames = 10
            self.processed_frames = i + 1
            time.sleep(0.5)

    def stop(self):
        self.running = False
