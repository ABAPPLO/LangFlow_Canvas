import hashlib
import os

import sqlalchemy as sa


DEFAULT_MIGRATION_LOCK_KEY = 11223344


def get_migration_lock_key() -> int:
    """Return the advisory lock key used to serialize migrations."""
    namespace = os.getenv("LANGFLOW_MIGRATION_LOCK_NAMESPACE")
    if not namespace:
        return DEFAULT_MIGRATION_LOCK_KEY
    return int(hashlib.sha256(namespace.encode()).hexdigest()[:16], 16) % (2**63 - 1)


def to_sync_database_url(database_url: str) -> str:
    """Convert an async-friendly database URL into a sync SQLAlchemy URL."""
    url = database_url
    if url.startswith("postgres://"):
        url = "postgresql://" + url.split("://", 1)[1]
    for async_driver in ("+asyncpg", "+aiosqlite"):
        url = url.replace(async_driver, "")
    return url


def table_exists(name, conn):
    """Check if a table exists.

    Parameters:
    name (str): The name of the table to check.
    conn (sqlalchemy.engine.Engine or sqlalchemy.engine.Connection): The SQLAlchemy engine or connection to use.

    Returns:
    bool: True if the table exists, False otherwise.
    """
    inspector = sa.inspect(conn)
    return name in inspector.get_table_names()


def column_exists(table_name, column_name, conn):
    """Check if a column exists in a table.

    Parameters:
    table_name (str): The name of the table to check.
    column_name (str): The name of the column to check.
    conn (sqlalchemy.engine.Engine or sqlalchemy.engine.Connection): The SQLAlchemy engine or connection to use.

    Returns:
    bool: True if the column exists, False otherwise.
    """
    inspector = sa.inspect(conn)
    return column_name in [column["name"] for column in inspector.get_columns(table_name)]


def foreign_key_exists(table_name, fk_name, conn):
    """Check if a foreign key exists in a table.

    Parameters:
    table_name (str): The name of the table to check.
    fk_name (str): The name of the foreign key to check.
    conn (sqlalchemy.engine.Engine or sqlalchemy.engine.Connection): The SQLAlchemy engine or connection to use.

    Returns:
    bool: True if the foreign key exists, False otherwise.
    """
    inspector = sa.inspect(conn)
    return fk_name in [fk["name"] for fk in inspector.get_foreign_keys(table_name)]


def constraint_exists(table_name, constraint_name, conn):
    """Check if a constraint exists in a table.

    Parameters:
    table_name (str): The name of the table to check.
    constraint_name (str): The name of the constraint to check.
    conn (sqlalchemy.engine.Engine or sqlalchemy.engine.Connection): The SQLAlchemy engine or connection to use.

    Returns:
    bool: True if the constraint exists, False otherwise.
    """
    inspector = sa.inspect(conn)
    constraints = inspector.get_unique_constraints(table_name)
    return constraint_name in [constraint["name"] for constraint in constraints]
