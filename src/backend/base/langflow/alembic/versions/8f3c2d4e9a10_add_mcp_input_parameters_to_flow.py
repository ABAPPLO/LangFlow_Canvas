"""add mcp input parameters to flow

Revision ID: 8f3c2d4e9a10
Revises: 077ae7b79634
Create Date: 2026-05-20 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from langflow.utils import migration

# revision identifiers, used by Alembic.
revision: str = "8f3c2d4e9a10"
down_revision: str | None = "077ae7b79634"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    conn = op.get_bind()
    with op.batch_alter_table("flow", schema=None) as batch_op:
        if not migration.column_exists(table_name="flow", column_name="mcp_input_parameters", conn=conn):
            batch_op.add_column(sa.Column("mcp_input_parameters", sa.JSON(), nullable=True))


def downgrade() -> None:
    conn = op.get_bind()
    with op.batch_alter_table("flow", schema=None) as batch_op:
        if migration.column_exists(table_name="flow", column_name="mcp_input_parameters", conn=conn):
            batch_op.drop_column("mcp_input_parameters")
