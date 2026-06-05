import queue
import time
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from detector import DetectorThread
from classifier import VLMWorker
from flywheel import DataFlywheel
from utils import compute_batch_summary
from report import build_dataframe, build_summary_text

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Recycling Intelligence",
    page_icon="♻",
    layout="wide",
)

st.title("♻ Vision-Based Recycling Intelligence")
st.caption("Hybrid YOLO + VLM pipeline · Phase 1 prototype")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Input")

    input_type = st.radio(
        "Source type",
        ["Image file", "Image folder", "Video file", "Mock (no file)"],
        help="Image file = single photo. Image folder = batch of photos. "
             "Video file = mp4/avi. Mock = simulated data, no file needed."
    )

    source_path = None

    if input_type == "Image file":
        uploaded = st.file_uploader(
            "Upload image", type=["jpg", "jpeg", "png", "bmp", "webp"]
        )
        if uploaded:
            tmp = Path("tmp_upload")
            tmp.mkdir(exist_ok=True)
            source_path = tmp / uploaded.name
            source_path.write_bytes(uploaded.getvalue())

    elif input_type == "Image folder":
        folder_input = st.text_input(
            "Folder path", value="assets/sample_images",
            help="Path to a folder containing jpg/png images"
        )
        source_path = Path(folder_input) if folder_input else None

    elif input_type == "Video file":
        uploaded_vid = st.file_uploader(
            "Upload video", type=["mp4", "avi", "mov", "mkv"]
        )
        if uploaded_vid:
            tmp = Path("tmp_upload")
            tmp.mkdir(exist_ok=True)
            source_path = tmp / uploaded_vid.name
            source_path.write_bytes(uploaded_vid.getvalue())
        else:
            path_input = st.text_input(
                "Or enter file path", value="assets/sample.mp4"
            )
            source_path = Path(path_input) if path_input else None

    else:
        source_path = None   # Mock mode

    st.divider()

    mode = st.radio(
        "VLM mode",
        ["mock (offline)", "real (OpenAI API)"],
        index=0,
        help="Mock runs instantly with no API cost. Use real for final demo.",
    )
    os.environ["VLM_MODE"] = "mock" if "mock" in mode else "openai"

    if "real" in mode:
        api_key = st.text_input("OpenAI API key", type="password")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key

    conf_threshold = st.slider(
        "Review flag threshold", 0.0, 1.0, 0.65, 0.05,
        help="Bottles below this confidence score are flagged for human review."
    )
    os.environ["CONFIDENCE_THRESHOLD"] = str(conf_threshold)

    st.divider()
    run_btn = st.button(
        "▶  Run analysis", type="primary", use_container_width=True,
        disabled=(st.session_state.get("phase") == "processing")
    )

# ── State machine: idle → processing → done ───────────────────────────────────
if "phase" not in st.session_state:
    st.session_state.phase = "idle"

# ── START ─────────────────────────────────────────────────────────────────────
if run_btn and st.session_state.phase in ("idle", "done"):
    # Clear any previous run
    for key in ["det", "vlm", "results", "latencies", "flywheel"]:
        st.session_state.pop(key, None)

    crop_q: queue.Queue = queue.Queue(maxsize=300)
    results: dict = {}
    latencies: list = []
    flywheel = DataFlywheel()

    det = DetectorThread(source_path, crop_q)
    vlm = VLMWorker(crop_q, results, latencies, flywheel=flywheel)
    det.start()
    vlm.start()

    st.session_state.update(
        phase="processing",
        det=det,
        vlm=vlm,
        results=results,
        latencies=latencies,
        flywheel=flywheel,
    )
    st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# PHASE: IDLE
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.phase == "idle":
    st.info("Configure your source in the sidebar, then click **▶ Run analysis**.")

    st.markdown("""
### What this system produces per bottle

| Deliverable | Description |
|---|---|
| **Bottle count** | Total unique objects detected (no double-counting) |
| **PET vs non-PET** | Material classification + per-bottle PET probability |
| **Color distribution** | Dominant color across the batch |
| **Confidence score** | Per-prediction confidence from the classifier |
| **Human-review flag** | Flagged when confidence < threshold |
| **Batch grade + value** | A/B/C/reject grade and estimated USD value |

### Tips
- Start with **Mock mode** to see all features instantly without any file
- Upload any photo of plastic bottles for image mode
- Switch source type freely between runs — click **▶ Run analysis** each time
- For pile images, tiled detection automatically activates for large photos
    """)

# ─────────────────────────────────────────────────────────────────────────────
# PHASE: PROCESSING
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.phase == "processing":
    det: DetectorThread = st.session_state.det
    vlm: VLMWorker = st.session_state.vlm
    results: dict = st.session_state.results
    latencies: list = st.session_state.latencies

    st.subheader("Processing…")

    col_feed, col_live = st.columns([3, 2])

    with col_feed:
        st.caption("Latest detected frame")
        frame_slot = st.empty()
        if det.frame is not None:
            frame_slot.image(det.frame, channels="BGR", use_column_width=True)
        else:
            frame_slot.info("Waiting for first frame…")

    with col_live:
        total = max(det.total_frames, 1)
        prog = min(det.processed_frames / total, 1.0)
        st.progress(prog, text=f"Frame {det.processed_frames} / {det.total_frames or '?'}")

        m1, m2 = st.columns(2)
        m1.metric("Bottles found", det.count)
        m2.metric("Classified so far", len(results))

        if latencies:
            avg_lat = sum(latencies) / len(latencies)
            st.metric("Avg classifier latency", f"{avg_lat:.0f} ms")

        classified = list(results.values())
        if classified:
            pet = sum(1 for r in classified if r.material == "PET")
            st.metric(
                "PET so far",
                f"{pet} / {len(classified)}",
                delta=f"{pet/len(classified):.0%}"
            )

        if det.count_method == "density":
            st.info(
                f"Dense pile detected. YOLO count: {det.count} · "
                f"Density estimate: ~{det.density_estimate}"
            )

    # Check if detector has finished
    if det.done:
        with st.spinner("Finalising remaining classifications…"):
            vlm.wait_until_done(detector_done=True, timeout=25.0)
        st.session_state.phase = "done"
        st.rerun()
    else:
        time.sleep(0.6)
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# PHASE: DONE — full deliverable report
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.phase == "done":
    det: DetectorThread = st.session_state.det
    results: dict = st.session_state.results
    latencies: list = st.session_state.latencies

    classified = list(results.values())
    summary = compute_batch_summary(classified)
    df = build_dataframe(classified)

    # ── Guard: nothing was classified ────────────────────────────────────────
    if not classified or df.empty:
        st.warning("Processing finished but no bottles were classified.")
        st.markdown("""
**Possible reasons:**
- No bottles visible in the source image/video
- Confidence threshold too high — try lowering it in the sidebar
- YOLO model still downloading — wait a moment and try again
- Image too small or blurry for detection

Try **Mock (no file)** mode to confirm the dashboard works correctly.
        """)
        if st.button("🔄 Try again"):
            st.session_state.phase = "idle"
            st.rerun()
        st.stop()

    # ── Success banner ────────────────────────────────────────────────────────
    col_banner, col_method = st.columns([3, 1])
    with col_banner:
        st.success(
            f"Analysis complete — **{det.count} bottles** detected, "
            f"**{len(classified)} classified**."
        )
    with col_method:
        if det.count_method == "density":
            st.info(f"Pile image: density estimate **~{det.density_estimate}** bottles total")

    # ─────────────────────────────────────────────────────────────────────────
    # DELIVERABLES — 6 KPI metrics across the top
    # ─────────────────────────────────────────────────────────────────────────
    st.subheader("Batch summary — all 6 deliverables")
    k1, k2, k3, k4, k5, k6 = st.columns(6)

    # Deliverable 1 — Object / bottle count
    count_display = det.count
    count_delta = None
    if det.count_method == "density":
        count_delta = f"~{det.density_estimate} est. total"
    k1.metric(
        "Bottle count",
        count_display,
        delta=count_delta,
        help="Unique bottles detected. No double-counting — ByteTrack used for video."
    )

    # Deliverable 2 — PET vs non-PET
    k2.metric(
        "PET composition",
        f"{summary.pet_count} / {summary.total_count}",
        delta=f"{summary.avg_pet_probability:.0%} avg probability",
        help="Estimated PET vs non-PET material composition"
    )

    # Deliverable 3 — Color / quality
    k3.metric(
        "Dominant color",
        summary.dominant_color.title(),
        help="Most common color across all classified bottles"
    )

    # Deliverable 4 — Confidence score
    k4.metric(
        "Avg confidence",
        f"{summary.avg_confidence:.0%}",
        help="Average per-prediction confidence score from the classifier"
    )

    # Deliverable 5 — Human-review flag
    flag_delta = "no action needed" if summary.flagged_for_review == 0 else "need review"
    k5.metric(
        "Flagged",
        summary.flagged_for_review,
        delta=flag_delta,
        delta_color="normal" if summary.flagged_for_review == 0 else "inverse",
        help="Bottles below confidence threshold, flagged for human review"
    )

    # Deliverable 6 — Batch grade + value
    k6.metric(
        "Batch grade",
        summary.batch_grade,
        delta=f"≈ ${summary.estimated_value_usd:.2f} USD",
        help="A=≥85% PET, B=≥65%, C=≥40%, reject=<40%"
    )

    st.divider()

    # ─────────────────────────────────────────────────────────────────────────
    # CHARTS
    # ─────────────────────────────────────────────────────────────────────────
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.subheader("Material composition")
        mat = df["material"].value_counts().reset_index()
        mat.columns = ["material", "count"]
        st.bar_chart(mat.set_index("material"))

    with col_b:
        st.subheader("Color distribution")
        col_dist = df["color"].value_counts().reset_index()
        col_dist.columns = ["color", "count"]
        st.bar_chart(col_dist.set_index("color"))

    with col_c:
        st.subheader("Batch grade distribution")
        grade = df["batch_grade"].value_counts().reset_index()
        grade.columns = ["grade", "count"]
        st.bar_chart(grade.set_index("grade"))

    # Confidence distribution
    st.subheader("Confidence score distribution")
    conf_hist = df["confidence"].round(1).value_counts().sort_index().rename("bottles")
    st.bar_chart(conf_hist)

    # ─────────────────────────────────────────────────────────────────────────
    # PER-BOTTLE TABLE
    # ─────────────────────────────────────────────────────────────────────────
    st.subheader("Per-bottle results")
    display_cols = [
        "object_id", "material", "is_PET", "pet_probability",
        "color", "confidence", "review_flag", "batch_grade", "reasoning"
    ]
    st.dataframe(
        df[display_cols],
        use_container_width=True,
        height=400,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # HUMAN REVIEW QUEUE
    # ─────────────────────────────────────────────────────────────────────────
    st.subheader("Human review queue")
    flagged_df = df[df["review_flag"]][[
        "object_id", "material", "color", "confidence", "reasoning", "batch_grade"
    ]]
    if flagged_df.empty:
        st.success("No items flagged for review.")
    else:
        st.warning(f"{len(flagged_df)} bottles need human review")
        st.dataframe(flagged_df, use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────────
    # LAST DETECTED FRAME
    # ─────────────────────────────────────────────────────────────────────────
    if det.frame is not None:
        st.subheader("Annotated detection result")
        st.image(det.frame, channels="BGR", use_column_width=True)
        if det.count_method == "density":
            st.caption(
                f"YOLO directly detected {det.count} bottles (bounding boxes above). "
                f"Density heuristic estimates ~{det.density_estimate} bottles total in the pile."
            )

    # ─────────────────────────────────────────────────────────────────────────
    # TEXT SUMMARY
    # ─────────────────────────────────────────────────────────────────────────
    st.subheader("Batch report")
    summary_text = build_summary_text(
        summary,
        total_input_frames=det.total_frames,
        yolo_count=det.count,
        density_estimate=det.density_estimate,
        count_method=det.count_method,
    )
    st.code(summary_text, language=None)

    # ─────────────────────────────────────────────────────────────────────────
    # EXPORTS
    # ─────────────────────────────────────────────────────────────────────────
    st.subheader("Export")
    exp1, exp2 = st.columns(2)

    with exp1:
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇  Download CSV report",
            data=csv_bytes,
            file_name="recycling_report.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with exp2:
        st.download_button(
            "⬇  Download text summary",
            data=summary_text.encode("utf-8"),
            file_name="batch_summary.txt",
            mime="text/plain",
            use_container_width=True,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # RESTART
    # ─────────────────────────────────────────────────────────────────────────
    st.divider()
    if st.button("🔄 Run another batch", use_container_width=True):
        st.session_state.phase = "idle"
        st.rerun()
