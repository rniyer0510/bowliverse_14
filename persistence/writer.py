from app.persistence.session import SessionLocal
from app.persistence.models import Player, Analysis
from app.common.logger import get_logger

logger = get_logger(__name__)


def write_analysis(
    result: dict,
    **kwargs,
):
    """
    Persist one analysis safely.

    Contract:
    - Persistence failure must NEVER break analysis
    - Must tolerate schema drift from orchestrator
    """

    file_path = kwargs.get("file_path")
    hand = kwargs.get("hand")
    bowler_type = kwargs.get("bowler_type")

    # If no DB-relevant data, skip quietly
    if not file_path:
        logger.warning("Persistence skipped: missing file_path")
        return

    db = SessionLocal()
    try:
        # For now: single implicit player
        player = db.query(Player).first()
        if not player:
            player = Player(
                handedness=hand,
                bowler_type=bowler_type,
            )
            db.add(player)
            db.flush()

        analysis = Analysis(
            player_id=player.id,
            video_path=file_path,
            result_json=result,
        )

        db.add(analysis)
        db.commit()

    except Exception as e:
        logger.warning(f"Persistence skipped: {e}")
        db.rollback()

    finally:
        db.close()

