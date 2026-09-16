from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


SOURCE_PACK_ROOT = (
    Path(__file__).parent
    / "tests"
    / "fixtures"
    / "evaluation_baseline"
    / "aa_rc_002"
)
MY_FILES_MANIFEST_ROOT = (
    Path(__file__).parent
    / "tests"
    / "fixtures"
    / "evaluation_baseline"
    / "aa_rc_002_my_files"
)
MANIFEST_NAME = "manifest.json"
EXPECTED_SCHEMA_VERSION = "aa-rc-002-source-pack.v1"
EXPECTED_MY_FILES_SCHEMA_VERSION = "aa-rc-002-my-files-manifest.v1"
EXPECTED_CASE_ID = "AA-RC-002"
EXPECTED_HASH_ALGORITHM = "SHA-256"
EXPECTED_MY_FILES_BINDING = "MY_FILES_OWNER"
EXPECTED_SUCCESS_CRITERIA = [
    "source_of_truth",
    "canonical_provenance",
    "source_identity",
    "version_hash_integrity",
    "conflict_handling",
    "primary_source_over_memory_or_summary",
]
_MARKER = "EVALUATION FIXTURE — NOT PRODUCTION USER DATA"


class EvaluationSourcePackError(ValueError):
    pass


def _safe_member_name(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise EvaluationSourcePackError("Source-pack member name is invalid.")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or path.name != value and value.startswith("/"):
        raise EvaluationSourcePackError("Source-pack member escapes its root.")
    if value.startswith("./") or value.endswith("/"):
        raise EvaluationSourcePackError("Source-pack member name is not canonical.")
    return value


def _member_path(root: Path, name: str) -> Path:
    root = root.resolve()
    path = (root / _safe_member_name(name)).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise EvaluationSourcePackError("Source-pack member escapes its root.") from error
    return path


def _canonical_payload(manifest: dict[str, Any]) -> dict[str, Any]:
    logical_sources: list[dict[str, Any]] = []
    for source in manifest["logical_sources"]:
        members = [
            {
                "name": member["name"],
                "sha256": member["sha256"],
            }
            for member in source["members"]
        ]
        logical_sources.append(
            {
                "logical_name": source["logical_name"],
                "kind": source["kind"],
                "members": members,
            }
        )
    return {
        "schema_version": manifest["schema_version"],
        "case_id": manifest["case_id"],
        "fixture_version": manifest["fixture_version"],
        "hash_algorithm": manifest["hash_algorithm"],
        "logical_sources": logical_sources,
    }


def canonical_source_pack_bytes(manifest: dict[str, Any]) -> bytes:
    return json.dumps(
        _canonical_payload(manifest),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def source_pack_sha256(manifest: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_source_pack_bytes(manifest)).hexdigest()


def _canonical_my_files_payload(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": manifest["schema_version"],
        "case_id": manifest["case_id"],
        "fixture_version": manifest["fixture_version"],
        "hash_algorithm": manifest["hash_algorithm"],
        "source_binding": manifest["source_binding"],
        "supersedes_source_pack_sha256": manifest[
            "supersedes_source_pack_sha256"
        ],
        "success_criteria": manifest["success_criteria"],
        "logical_sources": [
            {
                "logical_name": source["logical_name"],
                "kind": source["kind"],
                "source_id": source["source_id"],
                "source_version": source["source_version"],
                "original_filename": source["original_filename"],
                "members": [
                    {
                        "name": member["name"],
                        "sha256": member["sha256"],
                    }
                    for member in source["members"]
                ],
            }
            for source in manifest["logical_sources"]
        ],
    }


def my_files_manifest_bytes(manifest: dict[str, Any]) -> bytes:
    return json.dumps(
        _canonical_my_files_payload(manifest),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def my_files_manifest_sha256(manifest: dict[str, Any]) -> str:
    return hashlib.sha256(my_files_manifest_bytes(manifest)).hexdigest()


def load_aa_rc_002_my_files_manifest(
    root: Path = MY_FILES_MANIFEST_ROOT,
) -> dict[str, Any]:
    """Load the dated live-source binding without copying source bytes."""

    root = root.resolve()
    try:
        manifest = json.loads(
            (root / MANIFEST_NAME).read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise EvaluationSourcePackError(
            "MY_FILES source manifest cannot be read."
        ) from error
    if not isinstance(manifest, dict):
        raise EvaluationSourcePackError("MY_FILES source manifest must be an object.")
    for key in (
        "schema_version",
        "case_id",
        "fixture_version",
        "hash_algorithm",
        "source_binding",
        "supersedes_source_pack_sha256",
        "success_criteria",
        "logical_sources",
        "source_pack_sha256",
    ):
        if key not in manifest:
            raise EvaluationSourcePackError(f"MY_FILES manifest is missing {key}.")
    if manifest["schema_version"] != EXPECTED_MY_FILES_SCHEMA_VERSION:
        raise EvaluationSourcePackError("Unsupported MY_FILES manifest schema.")
    if manifest["case_id"] != EXPECTED_CASE_ID:
        raise EvaluationSourcePackError("MY_FILES manifest case ID is incorrect.")
    if manifest["hash_algorithm"] != EXPECTED_HASH_ALGORITHM:
        raise EvaluationSourcePackError("MY_FILES manifest hash algorithm is incorrect.")
    if manifest["source_binding"] != EXPECTED_MY_FILES_BINDING:
        raise EvaluationSourcePackError("MY_FILES manifest binding is incorrect.")
    if manifest["success_criteria"] != EXPECTED_SUCCESS_CRITERIA:
        raise EvaluationSourcePackError("MY_FILES success criteria were weakened.")
    superseded = manifest["supersedes_source_pack_sha256"]
    if (
        not isinstance(superseded, str)
        or len(superseded) != 64
        or any(char not in "0123456789abcdef" for char in superseded)
    ):
        raise EvaluationSourcePackError("Historical source-pack hash is invalid.")
    if not isinstance(manifest["logical_sources"], list) or not manifest[
        "logical_sources"
    ]:
        raise EvaluationSourcePackError("MY_FILES logical sources are required.")
    if my_files_manifest_sha256(manifest) != manifest["source_pack_sha256"]:
        raise EvaluationSourcePackError("MY_FILES manifest hash mismatch.")

    seen_source_ids: set[str] = set()
    seen_names: set[str] = set()
    for source in manifest["logical_sources"]:
        if not isinstance(source, dict):
            raise EvaluationSourcePackError("MY_FILES logical source is invalid.")
        logical_name = source.get("logical_name")
        kind = source.get("kind")
        source_id = source.get("source_id")
        source_version = source.get("source_version")
        original_filename = source.get("original_filename")
        members = source.get("members")
        if (
            not isinstance(logical_name, str)
            or kind != "single_source"
            or not isinstance(source_id, str)
            or not isinstance(source_version, int)
            or source_version < 1
            or not isinstance(original_filename, str)
            or not isinstance(members, list)
            or len(members) != 1
        ):
            raise EvaluationSourcePackError("MY_FILES source metadata is invalid.")
        if source_id in seen_source_ids or original_filename in seen_names:
            raise EvaluationSourcePackError("MY_FILES source is duplicated.")
        seen_source_ids.add(source_id)
        seen_names.add(original_filename)
        member = members[0]
        if (
            not isinstance(member, dict)
            or member.get("name") != original_filename
            or not isinstance(member.get("sha256"), str)
            or len(member["sha256"]) != 64
            or any(char not in "0123456789abcdef" for char in member["sha256"])
        ):
            raise EvaluationSourcePackError("MY_FILES source hash metadata is invalid.")
    return {
        "manifest": manifest,
        "source_pack_sha256": manifest["source_pack_sha256"],
    }


def load_aa_rc_002_source_pack(
    root: Path = SOURCE_PACK_ROOT,
) -> dict[str, Any]:
    root = root.resolve()
    manifest_path = root / MANIFEST_NAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvaluationSourcePackError("Source-pack manifest cannot be read.") from error
    if not isinstance(manifest, dict):
        raise EvaluationSourcePackError("Source-pack manifest must be an object.")
    for key in (
        "schema_version",
        "case_id",
        "fixture_version",
        "hash_algorithm",
        "logical_sources",
        "source_pack_sha256",
    ):
        if key not in manifest:
            raise EvaluationSourcePackError(f"Manifest is missing {key}.")
    if manifest["schema_version"] != EXPECTED_SCHEMA_VERSION:
        raise EvaluationSourcePackError("Unsupported source-pack schema version.")
    if manifest["case_id"] != EXPECTED_CASE_ID:
        raise EvaluationSourcePackError("Source-pack case ID is incorrect.")
    if manifest["hash_algorithm"] != EXPECTED_HASH_ALGORITHM:
        raise EvaluationSourcePackError("Source-pack hash algorithm is incorrect.")
    if not isinstance(manifest["logical_sources"], list):
        raise EvaluationSourcePackError("logical_sources must be a list.")
    if source_pack_sha256(manifest) != manifest["source_pack_sha256"]:
        raise EvaluationSourcePackError("Source-pack manifest hash mismatch.")

    resolved_sources: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for source in manifest["logical_sources"]:
        if not isinstance(source, dict):
            raise EvaluationSourcePackError("Logical source must be an object.")
        logical_name = source.get("logical_name")
        kind = source.get("kind")
        members = source.get("members")
        if not isinstance(logical_name, str) or kind not in {
            "single_source",
            "logical_collection",
        }:
            raise EvaluationSourcePackError("Logical source metadata is invalid.")
        if not isinstance(members, list) or not members:
            raise EvaluationSourcePackError("Logical source members are required.")
        resolved_members: list[dict[str, Any]] = []
        for member in members:
            if not isinstance(member, dict):
                raise EvaluationSourcePackError("Source-pack member must be an object.")
            name = _safe_member_name(member.get("name"))
            expected_hash = member.get("sha256")
            if (
                not isinstance(expected_hash, str)
                or len(expected_hash) != 64
                or any(char not in "0123456789abcdef" for char in expected_hash)
            ):
                raise EvaluationSourcePackError("Source-pack member hash is invalid.")
            if name in seen_names:
                raise EvaluationSourcePackError("Source-pack member is duplicated.")
            seen_names.add(name)
            path = _member_path(root, name)
            try:
                data = path.read_bytes()
            except OSError as error:
                raise EvaluationSourcePackError("Source-pack member is missing.") from error
            actual_hash = hashlib.sha256(data).hexdigest()
            if actual_hash != expected_hash:
                raise EvaluationSourcePackError("Source-pack member hash mismatch.")
            if _MARKER.encode("utf-8") not in data:
                raise EvaluationSourcePackError("Source-pack marker is missing.")
            resolved_members.append(
                {
                    "name": name,
                    "sha256": actual_hash,
                    "bytes": data,
                    "logical_name": logical_name,
                }
            )
        if kind == "single_source" and len(resolved_members) != 1:
            raise EvaluationSourcePackError("single_source must have one member.")
        resolved_sources.append(
            {
                "logical_name": logical_name,
                "kind": kind,
                "members": resolved_members,
            }
        )
    return {
        "manifest": manifest,
        "sources": resolved_sources,
        "source_pack_sha256": manifest["source_pack_sha256"],
    }


def resolve_aa_rc_002_sources(
    root: Path = SOURCE_PACK_ROOT,
) -> list[dict[str, Any]]:
    """Return evaluation-only source bytes with no fixture path in the result."""

    pack = load_aa_rc_002_source_pack(root)
    resolved: list[dict[str, Any]] = []
    for source in pack["sources"]:
        for member in source["members"]:
            resolved.append(
                {
                    "logical_name": source["logical_name"],
                    "kind": source["kind"],
                    "member_name": member["name"],
                    "sha256": member["sha256"],
                    "bytes": member["bytes"],
                }
            )
    return resolved