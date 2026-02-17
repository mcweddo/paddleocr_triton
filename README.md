# Project Objective

This repository packages an end‑to‑end OCR pipeline based on PaddleOCR (PP‑OCRv5) for deployment on NVIDIA Triton Inference Server. It provides an ensemble of models and Python preprocessing/postprocessing to perform text detection and text recognition on input images through a single Triton endpoint.

## What it does
- Uses ONNX exports of PP‑OCRv5 for text detection and recognition.
- Wraps image preprocessing, detection postprocessing (box extraction and cropping), and recognition postprocessing (CTC decoding with language dictionaries) in Triton Python backends.
- Composes everything into an `ensemble_model` so clients can send one image and receive:
  - Detected text boxes (coordinates)
  - Recognized text strings
  - Recognition confidence scores
- Provides Docker and docker‑compose configuration to run Triton with the included `model_repository`.

## Key components
- `triton_infer/model_repository/`
  - `text_detection/` – ONNX detection model.
  - `text_recognition/` – ONNX recognition model.
  - `detection_preprocessing/`, `detection_postprocessing/` – Python backends to prepare inputs and process detection outputs.
  - `recognition_postprocessing/` – Python backend to decode recognition outputs using provided dictionaries (e.g., `en_dict.txt`).
  - `ensemble_model/` – Triton ensemble that chains the above into a single pipeline.
- Docker artifacts (`Dockerfile`, `docker-compose.yml`) to launch Triton and serve the ensemble.
- Example client scripts in `triton_infer/workspace/` to call the server.

## In short
Deploy PP‑OCRv5 on Triton Inference Server as a production‑ready OCR service with integrated pre/post‑processing, accessible via one ensemble endpoint.
