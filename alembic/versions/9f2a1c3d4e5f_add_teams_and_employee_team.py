"""add teams and employee team relationship

Revision ID: 9f2a1c3d4e5f
Revises: f016d69ce600

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "9f2a1c3d4e5f"
down_revision: Union[str, None] = "4db9de92481c"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "teams",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("manager_id", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(["manager_id"], ["employees.id"]),
    )
    op.create_index("ix_teams_id", "teams", ["id"], unique=False)
    op.create_index("ix_teams_name", "teams", ["name"], unique=True)
    op.create_index("ix_teams_manager_id", "teams", ["manager_id"], unique=True)

    op.add_column("employees", sa.Column("team_id", sa.Integer(), nullable=True))
    op.create_index("ix_employees_team_id", "employees", ["team_id"], unique=False)
    op.create_foreign_key(
        "fk_employees_team_id_teams",
        "employees",
        "teams",
        ["team_id"],
        ["id"],
    )


def downgrade() -> None:
    op.drop_constraint("fk_employees_team_id_teams", "employees", type_="foreignkey")
    op.drop_index("ix_employees_team_id", table_name="employees")
    op.drop_column("employees", "team_id")
    op.drop_index("ix_teams_manager_id", table_name="teams")
    op.drop_index("ix_teams_name", table_name="teams")
    op.drop_index("ix_teams_id", table_name="teams")
    op.drop_table("teams")
