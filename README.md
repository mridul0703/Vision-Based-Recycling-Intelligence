# ♻️ Vision-Based Recycling Intelligence System

An AI-powered recycling analytics platform that combines object detection and vision-language models to analyze plastic bottle batches and generate operational insights for recycling facilities.

---

## Overview

This system automates bottle detection, material classification, contamination assessment, and reporting from images, videos, folders, or live streams.

The platform uses:

* **YOLOv8** for bottle detection
* **ByteTrack** for video tracking and counting
* **GPT-4o Vision** (or a local mock mode) for material classification
* **Streamlit** for visualization and reporting

It is designed to support recycling facilities by improving sorting efficiency, monitoring material streams, and reducing manual inspection effort.

---

## Features

* Multi-strategy bottle detection

  * Standard YOLO inference
  * Tiled detection for dense bottle piles
  * ByteTrack-based video tracking

* Vision-language classification

  * Material identification (PET, HDPE, PP, PVC)
  * Color classification
  * Contamination assessment

* Interactive dashboard

  * Detection statistics
  * Material distribution charts
  * Batch quality metrics
  * Real-time visualizations

* Automated reporting

  * CSV exports
  * Batch summary reports

* Data flywheel

  * Automatically stores high-confidence predictions
  * Creates a growing dataset for future local model training

---

## Architecture

```text
Input (Image / Video / Stream)
            │
            ▼
      YOLOv8 Detection
            │
            ▼
     Bottle Crop Extraction
            │
            ▼
     GPT-4o Classification
            │
            ▼
 Material & Quality Analysis
            │
            ├── Dataset Collection
            │
            ▼
   Dashboard & Report Generation
```

---

## Technology Stack

| Component       | Technology    |
| --------------- | ------------- |
| Frontend        | Streamlit     |
| Detection       | YOLOv8        |
| Tracking        | ByteTrack     |
| Classification  | GPT-4o Vision |
| Computer Vision | OpenCV        |
| Deep Learning   | PyTorch       |
| Data Processing | Pandas, NumPy |
| Validation      | Pydantic      |

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/<username>/recycling-intelligence.git
cd recycling-intelligence
```

### Create a Virtual Environment

```bash
python -m venv venv
```

Linux/macOS:

```bash
source venv/bin/activate
```

Windows:

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📦 Project Structure

### `main.py`

Streamlit-based dashboard that serves as the application's main interface. It manages user inputs, visualizations, processing status, review workflows, and report exports.

### `detector.py`

Handles bottle detection using YOLOv8. Supports standard image inference, tiled detection for dense bottle piles, and ByteTrack-based tracking for videos.

### `classifier.py`

Processes detected bottle crops using either GPT-4o Vision or a local mock classifier. Identifies material type, color, contamination level, and confidence score.

### `flywheel.py`

Implements the data collection pipeline by automatically storing labeled bottle crops and metadata for future model training.

### `schema.py`

Contains Pydantic models used for structured data validation and communication between different system components.

### `utils.py`

Provides utility functions for batch analysis, material statistics, grading logic, and value estimation.

### `report.py`

Generates structured outputs including CSV exports and human-readable batch summary reports.

### `mock_vlm.py`

Offline classification module used for testing and demonstrations when API access is unavailable.

### `.env`

Stores runtime configuration such as VLM mode, confidence thresholds, and API credentials.

---

## Configuration

Create a `.env` file in the project root:

```env
VLM_MODE=mock
CONFIDENCE_THRESHOLD=0.65
OPENAI_API_KEY=your_api_key
```

### Modes

| Mode   | Description                          |
| ------ | ------------------------------------ |
| mock   | Offline deterministic classification |
| openai | GPT-4o Vision classification         |

---

## Running the Application

```bash
streamlit run main.py
```

The dashboard will be available at:

```text
http://localhost:8501
```

---

## Dataset Structure

```text
dataset/
├── PET/
├── HDPE/
├── PP/
├── PVC/
└── unknown/
```

High-confidence predictions are automatically stored to support future local model training.

---

## Outputs

The system generates:

### CSV Reports

* Detection results
* Material labels
* Confidence scores
* Contamination assessments

### Batch Summaries

Example:

```text
Total Bottles: 482

PET: 61%
HDPE: 21%
PP: 11%
PVC: 3%
Unknown: 4%

Contamination Rate: 8%
Batch Grade: A
```

---
