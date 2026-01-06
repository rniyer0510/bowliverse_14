from fastapi import FastAPI, UploadFile, File, Form

from app.common.logger import get_logger
from app.io.loader import load_video

# Events
from app.workers.events.release_uah import detect_release_uah
from app.workers.events.ffc_bfc import detect_ffc_bfc

# Elbow (LOCKED – do not touch)
from app.workers.elbow.elbow_signal import compute_elbow_signal
from app.workers.elbow.elbow_legality import evaluate_elbow_legality

# Action (LOCKED)
from app.workers.action.action_classifier import classify_action

# Risk (v14 – cricket-native, baseball-style mechanics)
from app.workers.risk.risk_worker import run_risk_worker

# Interpretation (coach + child)
from app.interpretation.interpret_risks import interpret_risks

# Basic coaching diagnostics (NON-RISK)
from app.workers.efficiency.basic_coaching import analyze_basics


app = FastAPI()
logger = get_logger(__name__)


@app.post("/analyze")
def analyze(
    file: UploadFile = File(...),
    hand: str = Form(...),
    bowler_type: str = Form(None),  # DEPRECATED – retained for backward compatibility
):
    """
    ActionLab V14 – Orchestrator

    Pipeline:
    - load video + pose
    - detect Release → UAH
    - detect FFC → BFC
    - classify action (BFC-anchored)
    - compute risks (V14 injury/stress model)
    - compute basics (V14 coaching diagnostics, non-risk)
    - interpret risks
    - compute elbow legality (LOCKED)
    - assemble response
    """
    logger.info("Analyze request received")

    # Load
    video, pose_frames, _ = load_video(file)

    try:
        fps_val = float(video.get("fps") or 0.0)
    except Exception:
        fps_val = 0.0

    # Events: Release → UAH
    events = detect_release_uah(pose_frames=pose_frames, hand=hand, fps=fps_val)
    logger.info(f"Detected events (release/uah): {events}")

    # FFC/BFC anchored off UAH (ACTION anchoring)
    uah_frame = (events.get("uah") or {}).get("frame")
    if uah_frame is not None:
        foot_events = detect_ffc_bfc(pose_frames=pose_frames, hand=hand, uah_frame=uah_frame)
        if foot_events:
            events.update(foot_events)

    bfc_frame = (events.get("bfc") or {}).get("frame")
    ffc_frame = (events.get("ffc") or {}).get("frame")

    # Action
    action = classify_action(
        pose_frames=pose_frames,
        hand=hand,
        bfc_frame=bfc_frame,
        ffc_frame=ffc_frame,
    )

    # Risks
    risks = run_risk_worker(
        pose_frames=pose_frames,
        video=video,
        events=events,
        action=action,
    )

    # Basics (non-risk, always helpful for amateurs)
    basics = analyze_basics(
        pose_frames=pose_frames,
        hand=hand,
        events=events,
        action=action,
    )

    # Interpretation
    interpretation = interpret_risks(risks)

    # Elbow signal + legality (LOCKED)
    elbow_signal = compute_elbow_signal(pose_frames=pose_frames, hand=hand)

    elbow = evaluate_elbow_legality(
        elbow_signal=elbow_signal,
        events=events,
        fps=fps_val,
        pose_frames=pose_frames,  # IMPORTANT: enables event-driven legality without INCONCLUSIVE
    )

    return {
        "schema": "actionlab.v14",
        "input": {"hand": hand},
        "video": {"fps": video.get("fps"), "total_frames": video.get("total_frames")},
        "events": events,
        "elbow": elbow,
        "action": action,
        "risks": risks,
        "basics": basics,
        "interpretation": interpretation,
    }
