# AI Powered Personalized Fitness Coach, NLP + Computer Vision

This project is a local first AI fitness coach that combines:
1) Conversational AI for coaching style guidance and Q and A
2) A lightweight recommendation engine that suggests exercises from a dataset
3) Computer vision based form feedback using MediaPipe pose estimation on uploaded videos
4) Rep counting and range of motion tracking from a depth signal
5) Progress tracking stored locally in SQLite with charts

All processing is local. No videos are uploaded to the cloud.

## Features

Chat
Uses a Hugging Face DialoGPT model with a safety focused coaching prompt and profile personalization.

Recommendations
Uses scikit learn TF IDF plus NearestNeighbors to recommend exercises based on goals and focus.

Form check
Upload a squat video. The app estimates knee and hip angles, builds a depth signal, counts reps, computes range of motion, and produces feedback plus an optional annotated output video.

Rep counting and range of motion
The form check module builds a depth signal from hip and knee vertical movement, smooths it, detects top and bottom events, and estimates rep count.
It also computes range of motion from the depth signal and plots it in the UI.

Progress tracking
Log metrics and view trends inside the app.

Evaluation harness
Add your own short clips into eval_videos and label rep counts in data/eval_manifest.csv.
Run
python scripts/evaluate_pose_module.py
The script reports rep count MAE and accuracy within 1 rep and within 2 reps.

## Ethical and safety notes

This is not medical advice.
If you have pain, injury, or health concerns, consult a qualified professional.
The tool focuses on general technique cues, progressive overload basics, and safer defaults.

Privacy by design
Videos are processed on device. Session data is stored in a local SQLite database.

## Tech stack

Python
Streamlit UI
Transformers for NLP
MediaPipe plus OpenCV for pose detection and video processing
scikit learn for recommendations
SQLite for local sessions

## Setup, Windows

Recommended Python versions
Python 3.10 or 3.11 is the safest for MediaPipe on Windows.

1) Create venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

2) Install dependencies
pip install -r requirements.txt

3) Create .env
Copy .env.example to .env

4) Run
streamlit run app.py

## Demo flow for interviews

1) Set a profile, goals like build muscle
2) Ask for a plan and recommended exercises
3) Type check my form, then upload a squat video in the Form Check tab
4) Show rep count, depth chart, and the coach summary
5) Log progress points and show the chart

## Notes and limitations

Rep counting is an estimate from a 2D depth proxy and works best with side view and stable camera.
This is a portfolio project and not a medical device.

## Future upgrades

Better exercise specific detectors, deadlift, bench, overhead press
Rep segmentation table, tempo estimation, range of motion per rep
Larger exercise datasets and richer recommendation modeling
Optional voice input and speech output, still local
Auth and multi user profiles if deployed in a controlled environment
