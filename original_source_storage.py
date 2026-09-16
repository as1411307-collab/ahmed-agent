from __future__ import annotations

import hashlib
import re
from typing import Any, Protocol
from uuid import UUID

from persistence import (
    PersistenceError,
    create_original_source as _create_original_source,
    delete_original_source as _delete_original_source,
    read_authorized_original_source as _read_authorized_original_source,
    update_original_source_extraction_status as _update_original_source_extraction_status,
)


class OriginalSourceStore(Protocol):
    async def create(
        self,
        *,
        source_id: str,
        owner_principal_id: str,
        original_filename: str,
        mime_type: str | None,
        data: bytes,
        authorization_scope: str,
    ) -> dict[str, Any]:
        ...

    async def read(
        self,
        *,
        source_id: str,
        owner_principal_id: str | None,
    ) -> dict[str, Any]:
        ...

    async def set_extraction_status(
        self,
        *,
        source_id: str,
        status: str,
    ) -> None:
        ...

    async def delete(self, *, source_id: str) -> None:
        ...


_SOURCE_ID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-8][0-9a-fA-F]{3}-"
    r"[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}$"
)


def validate_source_id(value: object) -> str:
    if not isinstance(value, str) or not _SOURCE_ID_RE.fullmatch(value):
        raise ValueError("source_id must be a UUID")
    return str(UUID(value))


def source_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class PostgresOriginalSourceStore:
    """Durable PostgreSQL-backed implementation of the source-store contract."""

    async def create(
        self,
        *,
        source_id: str,
        owner_principal_id: str,
        original_filename: str,
        mime_type: str | None,
        data: bytes,
        authorization_scope: str,
    ) -> dict[str, Any]:
        return await _create_original_source(
            source_id=validate_source_id(source_id),
            owner_principal_id=owner_principal_id,
            original_filename=original_filename,
            mime_type=mime_type,
            data=data,
            authorization_scope=authorization_scope,
        )

    async def read(
        self,
        *,
        source_id: str,
        owner_principal_id: str | None,
    ) -> dict[str, Any]:
        return await _read_authorized_original_source(
            source_id=validate_source_id(source_id),
            owner_principal_id=owner_principal_id,
        )

    async def set_extraction_status(
        self,
        *,
        source_id: str,
        status: str,
    ) -> None:
        await _update_original_source_extraction_status(
            source_id=validate_source_id(source_id),
            status=status,
        )

    async def delete(self, *, source_id: str) -> None:
        await _delete_original_source(source_id=validate_source_id(source_id))


original_source_store: OriginalSourceStore = PostgresOriginalSourceStore()