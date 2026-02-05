from app.persistence.models import Account, Player, AccountPlayerLink
from app.common.logger import get_logger

logger = get_logger(__name__)

MAX_PARENT_PLAYERS = 3


def resolve_account(db, actor: dict):
    role = actor.get("role", "player").lower()
    name = actor.get("account_name", "Default")

    account = (
        db.query(Account)
        .filter_by(role=role, name=name)
        .first()
    )
    if account:
        return account

    account = Account(role=role, name=name)
    db.add(account)
    db.flush()

    logger.info(f"[identity] created account role={role} name={name}")
    return account


def resolve_player(db, account: Account, actor: dict):
    """
    Analysis-time resolver.

    IMPORTANT:
    - Does NOT override age_group / season for analysis
    - analysis_run MUST receive season & age_group explicitly
    - Player values are DEFAULTS only
    """

    role = account.role.lower()

    # Player self-account
    if role == "player":
        link = (
            db.query(AccountPlayerLink)
            .filter_by(account_id=account.account_id, link_type="self")
            .first()
        )
        if link:
            return link.player_id

        player = Player(
            primary_owner_account_id=account.account_id,
            created_by_account_id=account.account_id,
            age_group="ADULT",
            season=2025,
        )
        db.add(player)
        db.flush()

        db.add(AccountPlayerLink(
            account_id=account.account_id,
            player_id=player.player_id,
            link_type="self",
            player_name=actor.get("player_name"),
        ))

        return player.player_id

    # Coach / Parent

    if actor.get("player_id"):
        return actor["player_id"]

    player_name = actor.get("player_name")
    if not player_name:
        raise ValueError("player_name required for coach/parent uploads")

    existing = (
        db.query(AccountPlayerLink)
        .filter_by(
            account_id=account.account_id,
            player_name=player_name,
        )
        .first()
    )
    if existing:
        return existing.player_id

    if role == "parent":
        count = (
            db.query(AccountPlayerLink)
            .filter_by(account_id=account.account_id, link_type="child")
            .count()
        )
        if count >= MAX_PARENT_PLAYERS:
            raise ValueError("Parent player limit exceeded")

    link_type = "child" if role == "parent" else "trainee"

    player = Player(
        primary_owner_account_id=account.account_id,
        created_by_account_id=account.account_id,
        age_group="ADULT",
        season=2025,
    )
    db.add(player)
    db.flush()

    db.add(AccountPlayerLink(
        account_id=account.account_id,
        player_id=player.player_id,
        link_type=link_type,
        player_name=player_name,
    ))

    return player.player_id
