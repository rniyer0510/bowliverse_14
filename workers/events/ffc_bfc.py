from app.common.logger import get_logger
from app.workers.pose.landmarks import get_lm, vis_ok

logger = get_logger(__name__)

def _pair_ok(pf, a, b):
    return vis_ok(get_lm(pf, a)) and vis_ok(get_lm(pf, b))

def _score_pair(pf, a, b):
    return int(vis_ok(get_lm(pf, a))) + int(vis_ok(get_lm(pf, b)))

def detect_ffc_bfc(pose_frames, hand, uah_frame=None, ffc_frame=None, lookback=60):
    """
    ACTION-ONLY (reverse detection):
    - Pick FFC anchor as best-visibility FRONT-foot frame in [anchor-lookback .. anchor] (reverse window).
    - Pick BFC as best-visibility BACK-foot frame strictly before FFC in [ffc-lookback .. ffc-1].
    - Returns {} if insufficient visibility (honest output).
    """

    if ffc_frame is not None:
        anchor = int(ffc_frame)
    elif uah_frame is not None:
        anchor = int(uah_frame)
    else:
        return {}

    h = (hand or "R").upper()
    # Cricket: front-foot is the bowling-arm side foot at FFC, back-foot is the opposite at BFC.
    # R-hand: front=RIGHT, back=LEFT. L-hand: front=LEFT, back=RIGHT.
    if h == "R":
        front_heel, front_toe = "RIGHT_HEEL", "RIGHT_FOOT_INDEX"
        back_heel,  back_toe  = "LEFT_HEEL",  "LEFT_FOOT_INDEX"
    else:
        front_heel, front_toe = "LEFT_HEEL",  "LEFT_FOOT_INDEX"
        back_heel,  back_toe  = "RIGHT_HEEL", "RIGHT_FOOT_INDEX"

    start = max(0, anchor - int(lookback))
    logger.info(f"[FFC/BFC] anchor={anchor} lookback={lookback} window=[{start}..{anchor}]")

    # ------------------------------------------------------------
    # 1) Reverse-pick FFC anchor (best FRONT-foot visibility)
    # ------------------------------------------------------------
    best_ffc = None
    best_ffc_score = -1
    for i in range(anchor, start - 1, -1):
        s = _score_pair(pose_frames[i], front_heel, front_toe)
        if s > best_ffc_score:
            best_ffc_score = s
            best_ffc = i

    if best_ffc is None or best_ffc_score <= 0 or not _pair_ok(pose_frames[best_ffc], front_heel, front_toe):
        logger.info("[FFC/BFC] No usable FFC in window (front foot not visible enough)")
        return {}

    # ------------------------------------------------------------
    # 2) Reverse-pick BFC strictly BEFORE FFC (best BACK-foot visibility)
    # ------------------------------------------------------------
    bfc_anchor = best_ffc - 1
    if bfc_anchor <= 0:
        return {"ffc": {"frame": best_ffc}, "bfc": {"frame": None}}

    bfc_start = max(0, bfc_anchor - int(lookback))
    best_bfc = None
    best_bfc_score = -1

    for i in range(bfc_anchor, bfc_start - 1, -1):
        s = _score_pair(pose_frames[i], back_heel, back_toe)
        if s > best_bfc_score:
            best_bfc_score = s
            best_bfc = i

    if best_bfc is None or best_bfc_score <= 0 or not _pair_ok(pose_frames[best_bfc], back_heel, back_toe):
        logger.info("[FFC/BFC] No usable BFC before FFC (back foot not visible enough)")
        return {"ffc": {"frame": best_ffc}, "bfc": {"frame": None}}

    logger.info(f"[FFC/BFC] picked ffc={best_ffc} (score={best_ffc_score}) bfc={best_bfc} (score={best_bfc_score})")
    return {"ffc": {"frame": best_ffc}, "bfc": {"frame": best_bfc}}
