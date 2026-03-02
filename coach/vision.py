from __future__ import annotations

import os
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

import cv2
import mediapipe as mp


@dataclass
class FormFeedback:
    verdict: str
    details: List[str]
    stats: Dict[str, Any]
    annotated_video_path: Optional[str] = None


def _angle(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
    ax, ay = a
    bx, by = b
    cx, cy = c

    ab = (ax - bx, ay - by)
    cb = (cx - bx, cy - by)

    dot = ab[0] * cb[0] + ab[1] * cb[1]
    mag_ab = math.sqrt(ab[0] ** 2 + ab[1] ** 2)
    mag_cb = math.sqrt(cb[0] ** 2 + cb[1] ** 2)
    denom = max(1e-6, mag_ab * mag_cb)
    cosv = max(-1.0, min(1.0, dot / denom))
    return math.degrees(math.acos(cosv))


def _smooth(series: List[float], window: int = 7) -> List[float]:
    if window <= 1 or len(series) < window:
        return series[:]
    half = window // 2
    out: List[float] = []
    for i in range(len(series)):
        lo = max(0, i - half)
        hi = min(len(series), i + half + 1)
        out.append(sum(series[lo:hi]) / (hi - lo))
    return out


def _count_reps_from_depth(depth: List[float], min_frames_between: int = 8) -> Tuple[int, List[int], List[int]]:
    if len(depth) < 20:
        return 0, [], []

    peaks: List[int] = []
    valleys: List[int] = []

    for i in range(1, len(depth) - 1):
        if depth[i] > depth[i - 1] and depth[i] > depth[i + 1]:
            peaks.append(i)
        if depth[i] < depth[i - 1] and depth[i] < depth[i + 1]:
            valleys.append(i)

    if not peaks or not valleys:
        return 0, [], []

    dmin = min(depth)
    dmax = max(depth)
    amp = dmax - dmin
    if amp < 0.03:
        return 0, [], []

    peak_thr = dmin + amp * 0.65
    valley_thr = dmin + amp * 0.35

    peaks2 = [p for p in peaks if depth[p] >= peak_thr]
    valleys2 = [v for v in valleys if depth[v] <= valley_thr]

    reps = 0
    used_valleys: List[int] = []
    last_v = -10**9

    for v in valleys2:
        if v - last_v < min_frames_between:
            continue

        prev_peaks = [p for p in peaks2 if p < v]
        next_peaks = [p for p in peaks2 if p > v]

        if not prev_peaks or not next_peaks:
            continue

        prev_p = prev_peaks[-1]
        next_p = next_peaks[0]
        if (depth[prev_p] - depth[v]) < amp * 0.20:
            continue
        if (depth[next_p] - depth[v]) < amp * 0.20:
            continue

        reps += 1
        used_valleys.append(v)
        last_v = v

    return reps, used_valleys, peaks2


class PoseFormChecker:
    def __init__(self) -> None:
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils

    def analyze_squat_video(
        self,
        input_path: str,
        export_annotated: bool = True,
        out_dir: str = "outputs",
        min_frames: int = 25,
    ) -> FormFeedback:
        os.makedirs(out_dir, exist_ok=True)

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return FormFeedback(verdict="error", details=["Could not open video file."], stats={})

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        out_path = None
        writer = None
        if export_annotated and w and h:
            out_path = os.path.join(out_dir, "annotated_squat.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

        knee_angles: List[float] = []
        hip_angles: List[float] = []

        hip_y_series: List[float] = []
        knee_y_series: List[float] = []
        depth_series: List[float] = []

        frame_count = 0
        visible_frames = 0

        with self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as pose:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_count += 1

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = pose.process(rgb)

                if res.pose_landmarks:
                    visible_frames += 1
                    lm = res.pose_landmarks.landmark

                    def pt(idx: int) -> Tuple[float, float]:
                        return (lm[idx].x, lm[idx].y)

                    L_HIP = self.mp_pose.PoseLandmark.LEFT_HIP.value
                    L_KNEE = self.mp_pose.PoseLandmark.LEFT_KNEE.value
                    L_ANKLE = self.mp_pose.PoseLandmark.LEFT_ANKLE.value
                    L_SHOULDER = self.mp_pose.PoseLandmark.LEFT_SHOULDER.value

                    hip = pt(L_HIP)
                    knee = pt(L_KNEE)
                    ankle = pt(L_ANKLE)
                    shoulder = pt(L_SHOULDER)

                    knee_ang = _angle(hip, knee, ankle)
                    hip_ang = _angle(shoulder, hip, knee)

                    knee_angles.append(float(knee_ang))
                    hip_angles.append(float(hip_ang))

                    hip_y_series.append(float(hip[1]))
                    knee_y_series.append(float(knee[1]))
                    depth_series.append((float(hip[1]) + float(knee[1])) / 2.0)

                    if writer is not None:
                        self.mp_draw.draw_landmarks(frame, res.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                        cv2.putText(
                            frame,
                            f"knee {knee_ang:.1f} hip {hip_ang:.1f}",
                            (12, 28),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )

                if writer is not None:
                    writer.write(frame)

        cap.release()
        if writer is not None:
            writer.release()

        if frame_count < min_frames or visible_frames < max(10, int(min_frames * 0.4)):
            return FormFeedback(
                verdict="error",
                details=["Not enough clear pose frames. Use brighter video, full body in frame, side view for squats."],
                stats={"frames": frame_count, "pose_frames": visible_frames},
                annotated_video_path=out_path,
            )

        min_knee = min(knee_angles) if knee_angles else 180.0
        min_hip = min(hip_angles) if hip_angles else 180.0
        avg_knee = sum(knee_angles) / max(1, len(knee_angles))
        avg_hip = sum(hip_angles) / max(1, len(hip_angles))

        depth_sm = _smooth(depth_series, window=7)
        rep_count, valleys, peaks = _count_reps_from_depth(depth_sm, min_frames_between=int(max(6, fps * 0.2)))

        dmin = min(depth_sm) if depth_sm else 0.0
        dmax = max(depth_sm) if depth_sm else 0.0
        depth_range = float(dmax - dmin)

        stats: Dict[str, Any] = {
            "frames": frame_count,
            "pose_frames": visible_frames,
            "fps_est": float(fps),
            "knee_angle_min": float(min_knee),
            "knee_angle_avg": float(avg_knee),
            "hip_angle_min": float(min_hip),
            "hip_angle_avg": float(avg_hip),
            "rep_count_est": int(rep_count),
            "depth_min_norm": float(dmin),
            "depth_max_norm": float(dmax),
            "depth_range_norm": float(depth_range),
            "valley_frames": valleys,
            "peak_frames": peaks,
            "depth_series_norm": depth_sm,
        }

        details: List[str] = []
        verdict = "good"

        if min_knee > 120:
            verdict = "needs work"
            details.append("Depth looks shallow based on knee angle. If pain free, aim for a deeper position with control.")
        else:
            details.append("Depth looks decent based on knee angle.")

        if min_hip > 120:
            verdict = "needs work"
            details.append("Hip angle suggests limited hinge. Focus on bracing and sitting back, film from the side.")
        else:
            details.append("Hip angle suggests a reasonable squat position.")

        if rep_count == 0:
            verdict = "needs work" if verdict == "good" else verdict
            details.append("Rep count confidence is low. Keep full body in frame and stabilize the camera.")
        else:
            details.append(f"Estimated reps: {rep_count}. Range of motion signal captured.")

        if depth_range < 0.05:
            verdict = "needs work"
            details.append("Range of motion looks small. This can indicate shallow reps or camera angle issues.")
        else:
            details.append("Range of motion looks measurable from the depth signal.")

        details.append("General cues: brace, keep mid foot pressure, control descent, drive up with steady tempo.")

        return FormFeedback(verdict=verdict, details=details, stats=stats, annotated_video_path=out_path)
