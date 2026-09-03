"""add knowledge base tables

Revision ID: 4db9de92481c
Revises: f016d69ce600
Create Date: 2026-08-19 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '4db9de92481c'
down_revision: Union[str, None] = 'f016d69ce600'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table('knowledge_documents',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('title', sa.String(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_knowledge_documents_id'), 'knowledge_documents', ['id'], unique=False)
    op.create_index(op.f('ix_knowledge_documents_title'), 'knowledge_documents', ['title'], unique=False)

    op.create_table('knowledge_chunks',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('document_id', sa.Integer(), nullable=True),
    sa.Column('chunk_index', sa.Integer(), nullable=False),
    sa.Column('content', sa.Text(), nullable=False),
    sa.ForeignKeyConstraint(['document_id'], ['knowledge_documents.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_knowledge_chunks_id'), 'knowledge_chunks', ['id'], unique=False)
    op.create_index(op.f('ix_knowledge_chunks_document_id'), 'knowledge_chunks', ['document_id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_knowledge_chunks_document_id'), table_name='knowledge_chunks')
    op.drop_index(op.f('ix_knowledge_chunks_id'), table_name='knowledge_chunks')
    op.drop_table('knowledge_chunks')

    op.drop_index(op.f('ix_knowledge_documents_title'), table_name='knowledge_documents')
    op.drop_index(op.f('ix_knowledge_documents_id'), table_name='knowledge_documents')
    op.drop_table('knowledge_documents')
