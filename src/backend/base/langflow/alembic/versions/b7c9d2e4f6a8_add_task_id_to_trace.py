"""add task id to trace

Revision ID: b7c9d2e4f6a8
Revises: 8f3c2d4e9a10
Create Date: 2026-05-22 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
import sqlmodel
from alembic import op
from langflow.utils import migration

# revision identifiers, used by Alembic.
revision: str = "b7c9d2e4f6a8"
down_revision: str | None = "8f3c2d4e9a10"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _index_exists(table_name: str, index_name: str, conn) -> bool:
    inspector = sa.inspect(conn)
    return index_name in [index["name"] for index in inspector.get_indexes(table_name)]


def upgrade() -> None:
    conn = op.get_bind()
    if not migration.table_exists("trace", conn):
        return

    with op.batch_alter_table("trace", schema=None) as batch_op:
        if not migration.column_exists(table_name="trace", column_name="task_id", conn=conn):
            batch_op.add_column(sa.Column("task_id", sqlmodel.sql.sqltypes.AutoString(), nullable=True))
        if not _index_exists("trace", "ix_trace_task_id", conn):
            batch_op.create_index(batch_op.f("ix_trace_task_id"), ["task_id"], unique=False)


def downgrade() -> None:
    conn = op.get_bind()
    if not migration.table_exists("trace", conn):
        return

    with op.batch_alter_table("trace", schema=None) as batch_op:
        if _index_exists("trace", "ix_trace_task_id", conn):
            batch_op.drop_index(batch_op.f("ix_trace_task_id"))
        if migration.column_exists(table_name="trace", column_name="task_id", conn=conn):
            batch_op.drop_column("task_id")
