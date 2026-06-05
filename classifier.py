import base64
import json
import os
import queue
import threading
import time

import cv2
import openai
from dotenv import load_dotenv

from schema import BottleAnalysis
from mock_vlm import mock_classify

load_dotenv()

VLM_MODE = os.getenv("VLM_MODE", "mock")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.65"))

SYSTEM_PROMPT = """You are a recycling material classifier AI embedded in an industrial facility.
You receive cropped images of plastic bottles from a conveyor belt or batch image.
You must respond ONLY with a valid JSON object — no markdown, no explanation, no extra keys.

JSON schema (follow exactly):
{
  "material": "PET" | "HDPE" | "PP" | "PVC" | "unknown",
  "color": "<simple color name: clear/green/blue/amber/white/red/mixed/unknown>",
  "pet_probability": <float 0.0-1.0>,
  "confidence": <float 0.0-1.0, your self-assessed confidence in this classification>,
  "batch_grade": "A" | "B" | "C" | "reject",
  "reasoning": "<one sentence explanation>"
}

Grade guide: A = pure PET stream, B = recyclable non-PET, C = uncertain, reject = PVC or contaminated."""


def encode_crop(crop_bgr) -> str:
    _, buf = cv2.imencode(".jpg", crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode("utf-8")


def classify_real(obj_id: int, crop_bgr, frame_number: int = 0) -> BottleAnalysis:
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    b64 = encode_crop(crop_bgr)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=200,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64}",
                            "detail": "low",
                        },
                    },
                    {"type": "text", "text": "Classify this bottle."},
                ],
            },
        ],
    )

    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    data = json.loads(raw)
    data["object_id"] = obj_id
    data["frame_number"] = frame_number
    data["review_flag"] = data.get("confidence", 1.0) < CONFIDENCE_THRESHOLD
    return BottleAnalysis(**data)


class VLMWorker(threading.Thread):
    """
    Background thread that drains the crop queue and classifies each crop.
    Writes results into the shared `results` dict keyed by object_id.
    Tracks latencies for dashboard metrics.
    Calls flywheel.save() for high-confidence crops (dataset bootstrapping).
    """

    def __init__(self, crop_queue: queue.Queue, results: dict,
                 latencies: list, flywheel=None):
        super().__init__(daemon=True)
        self.crop_queue = crop_queue
        self.results = results
        self.latencies = latencies
        self.flywheel = flywheel
        self._stop_event = threading.Event()
        self._idle = threading.Event()

    def run(self):
        while not self._stop_event.is_set():
            try:
                item = self.crop_queue.get(timeout=0.5)
                self._idle.clear()
            except queue.Empty:
                self._idle.set()
                continue

            obj_id, crop, frame_number = item
            t0 = time.perf_counter()

            try:
                if VLM_MODE == "mock" or crop is None:
                    result = mock_classify(obj_id, frame_number)
                else:
                    result = classify_real(obj_id, crop, frame_number)
            except Exception as e:
                print(f"[VLMWorker] Error obj {obj_id}: {e}")
                result = mock_classify(obj_id, frame_number)

            self.results[obj_id] = result
            self.latencies.append(round((time.perf_counter() - t0) * 1000, 1))

            if self.flywheel and crop is not None:
                self.flywheel.save(obj_id, crop, result)

    def wait_until_done(self, detector_done: bool, timeout: float = 30.0) -> bool:
        """
        Blocks until the crop queue is fully drained AND the detector has
        finished. Called by the dashboard after det.done = True to ensure
        every detected bottle has been classified before showing the report.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            if detector_done and self.crop_queue.empty() and self._idle.is_set():
                return True
            time.sleep(0.2)
        return False   # timeout — show whatever we have

    def stop(self):
        self._stop_event.set()
