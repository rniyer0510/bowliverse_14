# PATCH FILE – applied manually into models.py

from sqlalchemy import CheckConstraint, Integer

# Add these fields inside class Player(Base):

age_group: Mapped[str] = mapped_column(
    String,
    nullable=False,
    default="ADULT",
)

season: Mapped[int] = mapped_column(
    Integer,
    nullable=False,
    default=2025,
)

__table_args__ = (
    CheckConstraint(
        "age_group IN ('U10','U14','U16','U19','ADULT')",
        name="ck_player_age_group",
    ),
)
