#!/usr/bin/env python3
"""CaseHub-backed CaseLoader for the Ark platform adapter."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any

from platform_client import (
    PlatformAPIError,
    TrainingPlatformClient,
    extract_case_detail,
    extract_case_items,
)

from openviking.session.train import Case, Rubric, RubricCriterion

_PROMPT_KEYS = (
    "input_prompt",
    "user_query",
    "prompt",
    "query",
    "instruction",
    "评测case",
)
_EXPECTED_ANSWER_KEYS = (
    "expected_answer",
    "ground_truth",
    "reference_answer",
    "answer",
    "正确答案",
)


@dataclass(slots=True)
class ArkCaseRepository:
    """Fetch, normalize and cache the cases frozen for a platform task."""

    client: TrainingPlatformClient
    dataset_ids: list[str]
    case_ids: list[str] = field(default_factory=list)
    caseset_id: str | None = None
    split_field: str | None = None
    page_size: int = 500
    detail_concurrency: int = 20
    default_split: str = "train"
    _cases: list[Case] | None = field(default=None, init=False, repr=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        self.dataset_ids = [str(item).strip() for item in self.dataset_ids if str(item).strip()]
        self.case_ids = list(
            dict.fromkeys(str(item).strip() for item in self.case_ids if str(item).strip())
        )
        if not self.dataset_ids and not self.caseset_id:
            raise ValueError("at least one dataset_id or caseset_id is required")
        if self.page_size <= 0:
            raise ValueError("page_size must be > 0")
        if self.detail_concurrency <= 0:
            raise ValueError("detail_concurrency must be > 0")
        self.split_field = str(self.split_field or "").strip() or None
        self.default_split = str(self.default_split or "train").strip().lower()

    async def cases_for_split(self, split: str) -> list[Case]:
        cases = await self.all_cases()
        normalized_split = _normalize_split(split)
        if self.split_field is None:
            return list(cases) if normalized_split == self.default_split else []
        return [
            case
            for case in cases
            if _normalize_split(_get_path(case.metadata, "platform_split")) == normalized_split
        ]

    async def all_cases(self) -> list[Case]:
        if self._cases is not None:
            return list(self._cases)
        async with self._lock:
            if self._cases is None:
                rows = await self._list_all_case_rows()
                details = await self._load_details(rows)
                normalized = [
                    _case_from_detail(detail, split_field=self.split_field) for detail in details
                ]
                if not normalized:
                    raise PlatformAPIError("CaseHub query returned no usable cases")
                self._cases = normalized
        return list(self._cases)

    async def _list_all_case_rows(self) -> list[dict[str, Any]]:
        if self.case_ids:
            return [{"case_id": case_id} for case_id in self.case_ids]

        scopes: list[tuple[str | None, str | None]]
        if self.caseset_id:
            scopes = [(self.caseset_id, None)]
        else:
            scopes = [(None, dataset_id) for dataset_id in self.dataset_ids]

        rows_by_id: dict[str, dict[str, Any]] = {}
        for caseset_id, dataset_id in scopes:
            offset = 0
            while True:
                payload = await self.client.list_cases(
                    caseset_id=caseset_id,
                    dataset_id=dataset_id,
                    limit=self.page_size,
                    offset=offset,
                )
                page = extract_case_items(payload)
                if not page:
                    break
                new_count = 0
                for row in page:
                    case_id = _case_id(row)
                    if case_id not in rows_by_id:
                        new_count += 1
                    rows_by_id[case_id] = row
                if len(page) < self.page_size or new_count == 0:
                    break
                offset += len(page)
        return list(rows_by_id.values())

    async def _load_details(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        semaphore = asyncio.Semaphore(self.detail_concurrency)

        async def load(row: dict[str, Any]) -> dict[str, Any]:
            case_id = _case_id(row)
            async with semaphore:
                payload = await self.client.get_case(case_id)
            detail = extract_case_detail(payload)
            return {**row, **detail, "case_id": case_id}

        return list(await asyncio.gather(*(load(row) for row in rows)))


@dataclass(slots=True)
class VikingCaseRepository:
    """Load the phase-frozen source cases of ark_viking_external_training."""

    client: TrainingPlatformClient
    task_id: str
    case_ids: list[str] = field(default_factory=list)
    page_size: int = 500
    _cases_by_phase: dict[str, list[Case]] = field(default_factory=dict, init=False, repr=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        self.task_id = str(self.task_id or "").strip()
        if not self.task_id:
            raise ValueError("task_id is required")
        self.case_ids = list(
            dict.fromkeys(str(item).strip() for item in self.case_ids if str(item).strip())
        )
        if self.page_size <= 0 or self.page_size > 500:
            raise ValueError("page_size must be between 1 and 500")

    async def cases_for_split(self, split: str) -> list[Case]:
        phase = _viking_phase_from_split(split)
        cached = self._cases_by_phase.get(phase)
        if cached is not None:
            return list(cached)
        async with self._lock:
            cached = self._cases_by_phase.get(phase)
            if cached is None:
                rows = await self._list_phase_rows(phase)
                requested = set(self.case_ids)
                if requested:
                    rows = [row for row in rows if _case_id(row) in requested]
                cached = [_case_from_viking_source(row, phase=phase) for row in rows]
                self._cases_by_phase[phase] = cached
        return list(cached)

    async def _list_phase_rows(self, phase: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        page = 1
        while True:
            payload = await self.client.list_rollout_source_cases(
                self.task_id,
                phase=phase,
                page=page,
                page_size=self.page_size,
            )
            current = [dict(item) for item in payload.get("cases") or [] if isinstance(item, dict)]
            rows.extend(current)
            total = int(payload.get("total") or 0)
            if not current or len(rows) >= total:
                break
            page += 1
        return rows


@dataclass(slots=True)
class ArkCaseLoader:
    repository: ArkCaseRepository | VikingCaseRepository
    split: str
    task_indices: list[int] | None = None
    dataset_ids: list[str] | None = None

    async def batches(self, context: Any = None) -> AsyncIterator[list[Case]]:
        del context
        cases = await self.repository.cases_for_split(self.split)
        if self.dataset_ids:
            selected_dataset_ids = set(self.dataset_ids)
            cases = [
                case
                for case in cases
                if str(case.metadata.get("dataset_id") or "") in selected_dataset_ids
            ]
        if self.task_indices is not None:
            selected: list[Case] = []
            for index in self.task_indices:
                if index < 0 or index >= len(cases):
                    raise IndexError(
                        f"task index {index} is out of range for split {self.split!r} "
                        f"with {len(cases)} cases"
                    )
                selected.append(cases[index])
            cases = selected
        if cases:
            yield cases

    async def split_exists(self) -> bool:
        return bool(await self.repository.cases_for_split(self.split))


def task_indices_from_filters(filters: dict[str, Any]) -> list[int] | None:
    raw = filters.get("task_indices")
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise ValueError("task_indices filter must be a list")
    indices: list[int] = []
    for value in raw:
        index = int(value)
        if index < 0:
            raise ValueError("task index must be >= 0")
        if index not in indices:
            indices.append(index)
    return indices


def _case_from_detail(detail: dict[str, Any], *, split_field: str | None) -> Case:
    case_id = _case_id(detail)
    envelope = _extract_envelope(detail)
    prompt = _find_scalar(envelope, _PROMPT_KEYS)
    if prompt is None:
        prompt = _find_scalar(detail, _PROMPT_KEYS)
    if prompt is None:
        raise PlatformAPIError(f"CaseHub case {case_id} has no input_prompt/user_query")
    expected_answer = _find_scalar(envelope, _EXPECTED_ANSWER_KEYS)
    if expected_answer is None:
        expected_answer = _find_scalar(detail, _EXPECTED_ANSWER_KEYS)

    name = str(detail.get("name") or envelope.get("name") or case_id)
    platform_split = None
    if split_field:
        platform_split = _get_path(envelope, split_field)
        if platform_split is None:
            platform_split = _get_path(detail, split_field)

    metadata = {
        "platform_case_id": case_id,
        "platform_split": platform_split,
        "caseset_id": _optional_text(detail.get("caseset_id")),
        "dataset_id": _optional_text(detail.get("dataset_id")),
        "schema_id": _optional_text(detail.get("schema_id")),
        "envelope_tos_uri": _optional_text(detail.get("envelope_tos_uri")),
    }
    metadata = {key: value for key, value in metadata.items() if value is not None}
    return Case(
        name=name,
        task_signature=_task_signature(detail),
        input={
            "task_id": case_id,
            "user_query": str(prompt),
            **({"expected_answer": str(expected_answer)} if expected_answer is not None else {}),
        },
        rubric=_rubric_from_payload(envelope, expected_answer=expected_answer),
        metadata=metadata,
    )


def _case_from_viking_source(row: dict[str, Any], *, phase: str) -> Case:
    case_id = _case_id(row)
    input_payload = row.get("input")
    input_payload = dict(input_payload) if isinstance(input_payload, dict) else {}
    expected_payload = row.get("expected_output")
    expected_payload = dict(expected_payload) if isinstance(expected_payload, dict) else {}
    prompt = _find_scalar(input_payload, _PROMPT_KEYS)
    if prompt is None:
        prompt = _find_scalar(row, _PROMPT_KEYS)
    if prompt is None:
        raise PlatformAPIError(f"Viking source case {case_id} has no prompt")
    expected_answer = row.get("expected_answer")
    if expected_answer is None:
        expected_answer = _find_scalar(expected_payload, _EXPECTED_ANSWER_KEYS)
    metadata = dict(row.get("metadata") or {})
    metadata.update(
        {
            "platform_case_id": case_id,
            "platform_split": phase,
            "viking_phase": phase,
        }
    )
    return Case(
        name=str(row.get("name") or case_id),
        task_signature=_task_signature(row),
        input={
            "task_id": case_id,
            "user_query": str(prompt),
            **({"expected_answer": str(expected_answer)} if expected_answer is not None else {}),
        },
        rubric=_rubric_from_payload(row, expected_answer=expected_answer),
        metadata=metadata,
    )


def _viking_phase_from_split(split: str) -> str:
    normalized = _normalize_split(split)
    return "train" if normalized == "train" else "validation"


def _rubric_from_payload(envelope: dict[str, Any], *, expected_answer: Any) -> Rubric:
    raw = envelope.get("rubric")
    if isinstance(raw, str):
        raw = _maybe_json(raw)
    if isinstance(raw, dict):
        criteria: list[RubricCriterion] = []
        for item in raw.get("criteria") or []:
            if not isinstance(item, dict):
                continue
            criteria.append(
                RubricCriterion(
                    name=str(item.get("name") or "platform_criterion"),
                    description=str(item.get("description") or ""),
                    required=bool(item.get("required", True)),
                    weight=float(item.get("weight", 1.0)),
                    metadata=dict(item.get("metadata") or {}),
                )
            )
        if criteria:
            return Rubric(
                name=str(raw.get("name") or "platform_evaluator"),
                description=str(raw.get("description") or ""),
                criteria=criteria,
                metadata=dict(raw.get("metadata") or {}),
            )
    description = "Use the platform rollout evaluator as the authoritative outcome signal."
    if expected_answer is not None:
        description += " The CaseHub envelope contains an expected answer."
    return Rubric(
        name="platform_rollout_builtin",
        description=description,
        criteria=[
            RubricCriterion(
                name="platform_rollout_builtin",
                description="Satisfy the platform-configured rollout evaluator.",
                required=True,
                weight=1.0,
            )
        ],
    )


def _extract_envelope(detail: dict[str, Any]) -> dict[str, Any]:
    for key in ("envelope", "case_envelope", "payload", "content"):
        raw = detail.get(key)
        parsed = _maybe_json(raw)
        if isinstance(parsed, dict):
            return parsed
    return detail


def _maybe_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text or text[0] not in "[{":
        return value
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def _find_scalar(value: Any, keys: tuple[str, ...]) -> Any | None:
    if isinstance(value, dict):
        for key in keys:
            candidate = value.get(key)
            if _is_scalar(candidate):
                return candidate
        for nested in value.values():
            found = _find_scalar(nested, keys)
            if found is not None:
                return found
    elif isinstance(value, list):
        for nested in value:
            found = _find_scalar(nested, keys)
            if found is not None:
                return found
    return None


def _get_path(value: Any, path: str) -> Any | None:
    current = value
    for part in str(path or "").split("."):
        if not part:
            continue
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _case_id(row: dict[str, Any]) -> str:
    for key in ("case_id", "id"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value)
    raise PlatformAPIError("CaseHub case row has no case_id")


def _task_signature(detail: dict[str, Any]) -> str:
    stable = json.dumps(detail, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"ark4-case:{sha256(stable.encode('utf-8')).hexdigest()}"


def _normalize_split(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    aliases = {
        "training": "train",
        "validation": "dev",
        "valid": "dev",
        "validation_seen": "dev",
        "valid_seen": "dev",
        "evaluation": "test",
        "eval": "test",
        "validation_unseen": "test",
        "valid_unseen": "test",
    }
    return aliases.get(text, text)


def _is_scalar(value: Any) -> bool:
    return value is not None and not isinstance(value, dict | list | tuple)


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
