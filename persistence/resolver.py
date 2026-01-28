from app.persistence.models import Account, Player, AccountPlayerLink
from app.common.logger import get_logger

logger = get_logger(__name__)

MAX_PARENT_PLAYERS = 3


def resolve_account(db, actor: dict):
    """
    mkdir -p semantics for Account
    Reuse if exists, create if missing.
    """
    role = actor.get("role", "PLAYER").upper()
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

    logger.info(
        f"[identity] created account role={role} name={name} id={account.account_id}"
    )
    return account


def resolve_player(db, account: Account, actor: dict):
    """
    mkdir -p semantics for Player resolution

    Rules:
    - PLAYER: exactly one self player
    - PARENT / COACH:
        * player_id wins if provided
        * else reuse by player_name if already linked
        * else create new (subject to limits)
    """

    # ======================================================
    # PLAYER → self
    # ======================================================
    if account.role == "PLAYER":
        link = (
            db.query(AccountPlayerLink)
            .filter_by(
                account_id=account.account_id,
                link_type="self",
            )
            .first()
        )
        if link:
            return link.player_id

        player = Player(
            primary_owner_account_id=account.account_id,
            created_by_account_id=account.account_id,
            name=actor.get("player_name"),
        )
        db.add(player)
        db.flush()

        db.add(
            AccountPlayerLink(
                account_id=account.account_id,
                player_id=player.player_id,
                link_type="self",
            )
        )

        logger.info(
            f"[identity] created self player id={player.player_id} for account={account.account_id}"
        )
        return player.player_id

    # ======================================================
    # PARENT / COACH
    # ======================================================

    # 1️⃣ Explicit player_id always wins
    if actor.get("player_id"):
        return actor["player_id"]

    player_name = actor.get("player_name")
    if not player_name:
        raise ValueError(
            "player_name or player_id required for parent/coach uploads"
        )

    # 2️⃣ Reuse existing linked player by name (mkdir -p)
    existing = (
        db.query(Player)
        .join(
            AccountPlayerLink,
            Player.player_id == AccountPlayerLink.player_id,
        )
        .filter(
            AccountPlayerLink.account_id == account.account_id,
            Player.name == player_name,
        )
        .first()
    )

    if existing:
        return existing.player_id

    # 3️⃣ Enforce parent limit BEFORE creation
    if account.role == "PARENT":
        count = (
            db.query(AccountPlayerLink)
            .filter_by(
                account_id=account.account_id,
                link_type="child",
            )
            .count()
        )
        if count >= MAX_PARENT_PLAYERS:
            raise ValueError("Parent cannot have more than 3 players")

    # 4️⃣ Create new player + link
    link_type = "child" if account.role == "PARENT" else "trainee"

    player = Player(
        primary_owner_account_id=account.account_id,
        created_by_account_id=account.account_id,
        name=player_name,
    )
    db.add(player)
    db.flush()

    db.add(
        AccountPlayerLink(
            account_id=account.account_id,
            player_id=player.player_id,
            link_type=link_type,
        )
    )

    logger.info(
        f"[identity] created player name={player_name} id={player.player_id} "
        f"linked_as={link_type} to account={account.account_id}"
    )

    return player.player_id

