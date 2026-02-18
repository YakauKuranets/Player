"""enterprise_schema

Revision ID: 0002_enterprise_schema
Revises: 0001_initial_schema
Create Date: 2026-02-18 00:30:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0002_enterprise_schema"
down_revision = "0001_initial_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    role_enum = sa.Enum("admin", "analyst", "viewer", name="userrole")
    role_enum.create(op.get_bind(), checkfirst=True)

    op.create_table(
        "teams",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(), nullable=False, unique=True),
        sa.Column("description", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
    )
    op.create_table(
        "workspaces",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("team_id", sa.Integer(), sa.ForeignKey("teams.id"), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=True),
    )
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("email", sa.String(), nullable=False, unique=True),
        sa.Column("username", sa.String(), nullable=False, unique=True),
        sa.Column("password_hash", sa.String(), nullable=False),
        sa.Column("role", role_enum, nullable=False),
        sa.Column("team_id", sa.Integer(), sa.ForeignKey("teams.id"), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("last_login_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    op.create_table(
        "user_sessions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("token_hash", sa.String(), nullable=False, unique=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("expires_at", sa.DateTime(), nullable=False),
        sa.Column("revoked", sa.Boolean(), nullable=True),
        sa.Column("ip_address", sa.String(), nullable=True),
        sa.Column("user_agent", sa.String(), nullable=True),
    )

    op.create_table(
        "enterprise_audit_logs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("timestamp", sa.DateTime(), nullable=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=True),
        sa.Column("team_id", sa.Integer(), sa.ForeignKey("teams.id"), nullable=True),
        sa.Column("action", sa.String(), nullable=False),
        sa.Column("resource_type", sa.String(), nullable=True),
        sa.Column("resource_id", sa.String(), nullable=True),
        sa.Column("details", sa.String(), nullable=True),
        sa.Column("ip_address", sa.String(), nullable=True),
        sa.Column("request_id", sa.String(), nullable=True),
        sa.Column("status", sa.String(), nullable=True),
    )
    op.create_index("ix_enterprise_audit_logs_timestamp", "enterprise_audit_logs", ["timestamp"], unique=False)
    op.create_index("ix_enterprise_audit_logs_request_id", "enterprise_audit_logs", ["request_id"], unique=False)

    op.add_column("cases", sa.Column("workspace_id", sa.Integer(), nullable=True))
    op.create_foreign_key("fk_cases_workspace_id", "cases", "workspaces", ["workspace_id"], ["id"])


def downgrade() -> None:
    op.drop_constraint("fk_cases_workspace_id", "cases", type_="foreignkey")
    op.drop_column("cases", "workspace_id")

    op.drop_index("ix_enterprise_audit_logs_request_id", table_name="enterprise_audit_logs")
    op.drop_index("ix_enterprise_audit_logs_timestamp", table_name="enterprise_audit_logs")
    op.drop_table("enterprise_audit_logs")
    op.drop_table("user_sessions")
    op.drop_index("ix_users_email", table_name="users")
    op.drop_table("users")
    op.drop_table("workspaces")
    op.drop_table("teams")

    sa.Enum(name="userrole").drop(op.get_bind(), checkfirst=True)
