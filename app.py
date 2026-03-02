from __future__ import annotations

import os
import time
from typing import Optional

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from coach.config import settings
from coach.db import Database
from coach.nlp import FitnessChatbot, UserProfile
from coach.recommender import ExerciseRecommender
from coach.vision import PoseFormChecker
from coach.utils import sanitize_text, detect_intent


st.set_page_config(page_title="AI Fitness Coach", layout="wide")

os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

db = Database(settings.db_path)


@st.cache_resource
def load_chatbot():
    return FitnessChatbot(settings.dialogpt_model)


@st.cache_resource
def load_recommender():
    return ExerciseRecommender("data/exercises.csv")


@st.cache_resource
def load_vision():
    return PoseFormChecker()


bot = load_chatbot()
rec = load_recommender()
vision = load_vision()


def get_session_id() -> int:
    if "session_id" not in st.session_state:
        st.session_state.session_id = db.create_session()
    return int(st.session_state.session_id)


def profile_ui(session_id: int) -> UserProfile:
    prof = db.get_profile(session_id)
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        age_raw = st.text_input("Age", value=str(prof.get("age") or ""))
        age: Optional[int] = None
        try:
            if age_raw.strip():
                age = int(age_raw.strip())
        except Exception:
            age = None

    with col2:
        exp_val = str(prof.get("experience") or "beginner")
        experience = st.selectbox(
            "Experience",
            ["beginner", "intermediate", "advanced"],
            index=["beginner", "intermediate", "advanced"].index(exp_val),
        )

    with col3:
        goals = st.text_input("Goals", value=str(prof.get("goals") or "build muscle"))

    with col4:
        constraints = st.text_input("Constraints", value=str(prof.get("constraints") or "none"))

    if st.button("Save profile"):
        db.upsert_profile(session_id=session_id, age=age, experience=experience, goals=goals, constraints=constraints)
        st.success("Profile saved locally.")

    prof2 = db.get_profile(session_id)
    return UserProfile(
        age=prof2.get("age", None),
        experience=str(prof2.get("experience") or "beginner"),
        goals=str(prof2.get("goals") or "build muscle"),
        constraints=str(prof2.get("constraints") or "none"),
    )


def render_chat(session_id: int, profile: UserProfile):
    st.subheader("Chat")

    messages = db.get_messages(session_id, limit=40)
    for m in messages:
        if m["role"] == "user":
            st.markdown(f"**You:** {m['content']}")
        else:
            st.markdown(f"**Coach:** {m['content']}")

    user_text = st.text_input("Message", value="", placeholder="Ask for a plan, exercise suggestions, or say check my form.")
    if st.button("Send"):
        user_text = sanitize_text(user_text)
        if not user_text:
            st.warning("Type a message first.")
            return

        db.add_message(session_id, "user", user_text)
        intent = detect_intent(user_text)

        extra = ""
        if intent["recommend"]:
            diff = profile.experience
            recs = rec.recommend(goals=profile.goals, focus="", difficulty=diff, top_k=6)
            if recs:
                lines = []
                for r in recs:
                    lines.append(f"{r.exercise_name} | {r.muscle_group} | {r.difficulty} | {r.reason}")
                extra = "Recommended exercises based on profile:\n" + "\n".join(lines)

        if intent["form_check"]:
            extra = (extra + "\n\n" if extra else "") + "User requested form check. Ask them to upload a squat video on the Form Check tab."

        short_context = db.get_messages(session_id, limit=20)
        reply = bot.generate(user_text=user_text, profile=profile, short_context=short_context, extra_context=extra)

        if intent["form_check"]:
            reply = reply + "\n\nGo to Form Check and upload a squat video. Side view, full body, stable camera, good lighting."

        db.add_message(session_id, "assistant", reply)
        st.rerun()


def plot_depth_series(depth_series: list[float], valleys: list[int], peaks: list[int]):
    if not depth_series:
        return
    fig = plt.figure()
    plt.plot(depth_series, label="depth signal")
    if peaks:
        plt.scatter(peaks, [depth_series[i] for i in peaks], marker="o", label="top positions")
    if valleys:
        plt.scatter(valleys, [depth_series[i] for i in valleys], marker="x", label="bottom positions")
    plt.xlabel("frame index")
    plt.ylabel("normalized depth (higher means deeper)")
    plt.title("Squat depth and rep events")
    plt.legend()
    st.pyplot(fig)


def render_form_check(session_id: int, profile: UserProfile):
    st.subheader("Form Check, squat focused")

    st.write("Privacy note: video is processed locally on your machine, nothing is uploaded to the cloud.")
    st.caption("Rep counting and ROM are estimated from the depth signal. Best results come from side view with full body in frame.")

    uploaded = st.file_uploader("Upload a short video", type=["mp4", "mov", "avi", "mkv"])
    if uploaded is None:
        st.info("Tip: side view, full body in frame, steady camera, good lighting.")
        return

    ts = int(time.time())
    in_path = os.path.join("uploads", f"upload_{ts}_{uploaded.name}")
    with open(in_path, "wb") as f:
        f.write(uploaded.read())

    if st.button("Analyze video"):
        with st.spinner("Analyzing form..."):
            feedback = vision.analyze_squat_video(in_path, export_annotated=True, out_dir="outputs")

        if feedback.verdict == "error":
            st.error("Could not analyze video.")
            for d in feedback.details:
                st.write(d)
            st.json(feedback.stats)
            return

        st.markdown(f"**Verdict:** {feedback.verdict}")

        rep_count = int(feedback.stats.get("rep_count_est", 0))
        depth_range = float(feedback.stats.get("depth_range_norm", 0.0))

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Estimated reps", rep_count)
        with c2:
            st.metric("ROM depth range", f"{depth_range:.3f}")
        with c3:
            st.metric("Pose frames", int(feedback.stats.get("pose_frames", 0)))

        st.subheader("Feedback")
        for d in feedback.details:
            st.write(f"• {d}")

        st.subheader("Depth and rep events")
        depth_series = feedback.stats.get("depth_series_norm", []) or []
        valleys = feedback.stats.get("valley_frames", []) or []
        peaks = feedback.stats.get("peak_frames", []) or []
        plot_depth_series(depth_series, valleys, peaks)

        st.subheader("Stats")
        st.json(feedback.stats)

        if feedback.annotated_video_path and os.path.exists(feedback.annotated_video_path):
            st.subheader("Annotated output")
            st.video(feedback.annotated_video_path)

        extra = (
            "Form check results:\n"
            + f"Verdict: {feedback.verdict}\n"
            + f"Estimated reps: {rep_count}\n"
            + f"Depth range of motion: {depth_range:.3f}\n"
            + "Details:\n"
            + "\n".join(feedback.details)
            + "\nStats:\n"
            + str(feedback.stats)
        )

        short_context = db.get_messages(session_id, limit=20)
        user_text = "Summarize my form check and give 3 cues for next session. Keep it short."
        reply = bot.generate(user_text=user_text, profile=profile, short_context=short_context, extra_context=extra)
        db.add_message(session_id, "assistant", reply)

        st.subheader("Coach summary")
        st.write(reply)

        st.caption("Evaluation harness: add clips to eval_videos and edit data/eval_manifest.csv, then run python scripts/evaluate_pose_module.py")


def render_progress(session_id: int):
    st.subheader("Progress tracking")

    col1, col2, col3 = st.columns(3)
    with col1:
        metric = st.selectbox("Metric", ["body_weight", "squat_1rm_est", "bench_1rm_est", "workouts_per_week", "steps_per_day"])
    with col2:
        value = st.number_input("Value", value=0.0, step=1.0)
    with col3:
        note = st.text_input("Note", value="")

    if st.button("Log progress"):
        db.add_progress(session_id, metric, float(value), note)
        st.success("Saved.")

    rows = db.get_progress(session_id)
    if not rows:
        st.info("Log a few entries to see charts.")
        return

    df = pd.DataFrame(rows)
    df["dt"] = pd.to_datetime(df["ts"], unit="s")
    st.dataframe(df[["dt", "metric", "value", "note"]], use_container_width=True)

    selected = st.selectbox("Chart metric", sorted(df["metric"].unique().tolist()))
    d2 = df[df["metric"] == selected].copy()

    fig = plt.figure()
    plt.plot(d2["dt"], d2["value"])
    plt.xlabel("date")
    plt.ylabel(selected)
    plt.title(f"{selected} over time")
    st.pyplot(fig)


def main():
    st.title("AI Powered Personalized Fitness Coach, NLP + Computer Vision")

    session_id = get_session_id()

    with st.expander("Profile", expanded=True):
        profile = profile_ui(session_id)

    tab1, tab2, tab3 = st.tabs(["Chat", "Form Check", "Progress"])
    with tab1:
        render_chat(session_id, profile)
    with tab2:
        render_form_check(session_id, profile)
    with tab3:
        render_progress(session_id)


if __name__ == "__main__":
    main()
