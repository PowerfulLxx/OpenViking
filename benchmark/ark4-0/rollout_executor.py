#!/usr/bin/env python3
"""RolloutExecutor translating gateway rollout results into OpenViking Rollouts."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any, Literal

import httpx
from platform_client import PlatformAPIError, TrainingPlatformClient

from openviking.message import ImagePart, Message, TextPart, ToolPart
from openviking.session.train import (
    Case,
    CriterionResult,
    ExecutionContext,
    ExperienceSet,
    Rollout,
    RubricEvaluation,
)

_TERMINAL_CASE_STATUSES = {"completed", "failed", "canceled", "cancelled"}
_ROLLOUT_BATCH_METADATA_KEY = "_ark_rollout_batch"


@dataclass(slots=True)
class RolloutBatchCoordinator:
    """Share one platform batch submission across per-case adapter calls."""

    _submissions: dict[str, asyncio.Task[dict[str, Any]]] = field(
        default_factory=dict,
        init=False,
    )
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    async def get_or_submit(
        self,
        key: str,
        factory: Callable[[], Awaitable[dict[str, Any]]],
    ) -> dict[str, Any]:
        async with self._lock:
            task = self._submissions.get(key)
            if task is None:
                task = asyncio.create_task(factory())
                self._submissions[key] = task
        return await asyncio.shield(task)


@dataclass(slots=True)
class ArkRolloutExecutor:
    client: TrainingPlatformClient
    platform_task_id: str
    connector_config: dict[str, Any] = field(default_factory=dict)
    runtime_params: dict[str, Any] = field(default_factory=dict)
    extra_header: dict[str, Any] = field(default_factory=dict)
    poll_interval_seconds: float = 2.0
    timeout_seconds: float = 3600.0
    idempotency_namespace: str = "openviking-ark4"
    require_messages_for_training: bool = True
    concurrency: int = 16
    batch_coordinator: RolloutBatchCoordinator | None = None

    def __post_init__(self) -> None:
        if not str(self.platform_task_id or "").strip():
            raise ValueError("platform_task_id is required")
        if self.poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be > 0")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")
        if self.concurrency <= 0:
            raise ValueError("concurrency must be > 0")

    async def execute(
        self,
        cases: list[Case],
        policy_set: ExperienceSet,
        context: ExecutionContext,
    ) -> list[Rollout]:
        del policy_set
        case_list = list(cases)
        phases = {
            str(case.metadata.get("viking_phase") or "").strip().lower()
            for case in case_list
        }
        if (
            len(case_list) > 1
            and len(phases) == 1
            and next(iter(phases)) in {"", "train", "validation"}
        ):
            return await self._execute_batch(case_list, context)
        semaphore = asyncio.Semaphore(self.concurrency)

        async def run_one(case: Case) -> Rollout:
            async with semaphore:
                return await self._execute_one(case, context)

        return list(await asyncio.gather(*(run_one(case) for case in case_list)))

    async def _execute_batch(
        self,
        cases: list[Case],
        context: ExecutionContext,
    ) -> list[Rollout]:
        phase = str(cases[0].metadata.get("viking_phase") or "").strip().lower()
        case_by_id: dict[str, Case] = {}
        for case in cases:
            case_id = str(
                case.metadata.get("platform_case_id")
                or case.input.get("task_id")
                or ""
            ).strip()
            if not case_id:
                raise ValueError(f"case {case.name!r} has no platform_case_id")
            if case_id in case_by_id:
                raise ValueError(f"duplicate platform_case_id: {case_id}")
            case_by_id[case_id] = case

        connector_config = _deep_merge(
            self.connector_config,
            {"options": {"repeat": 1}},
        )
        body: dict[str, Any] = {
            "case_ids": list(case_by_id),
            "workers": min(self.concurrency, len(case_by_id)),
            "connector_config": connector_config,
            "runtime_params": dict(self.runtime_params),
            "extra_header": dict(self.extra_header),
        }
        if phase:
            body["phase"] = phase
        idempotency_key = _idempotency_key(
            namespace=self.idempotency_namespace,
            task_id=self.platform_task_id,
            case=cases[0],
            context=context,
            body=body,
        )
        submission = await self.client.submit_rollout_eval(
            self.platform_task_id,
            body=body,
            idempotency_key=idempotency_key,
        )
        batch_rollout_id = str(submission.get("batch_rollout_id") or "")
        handles = {
            str(item.get("case_id") or ""): str(
                item.get("case_rollout_id") or ""
            )
            for item in submission.get("case_rollouts") or []
            if isinstance(item, dict)
        }
        missing = [case_id for case_id in case_by_id if not handles.get(case_id)]
        if missing:
            raise PlatformAPIError(
                "batch rollout returned no handle for: "
                + ", ".join(missing)
            )

        semaphore = asyncio.Semaphore(self.concurrency)

        async def poll_one(case_id: str, case: Case) -> Rollout:
            case_rollout_id = handles[case_id]
            async with semaphore:
                result = await self._poll(case_rollout_id, case=case)
            return _rollout_from_platform_result(
                case=case,
                result=result,
                policy_snapshot_id=context.policy_snapshot_id,
                platform_task_id=self.platform_task_id,
                batch_rollout_id=batch_rollout_id,
                case_rollout_id=case_rollout_id,
                require_messages=(
                    self.require_messages_for_training
                    and bool(context.metadata.get("training"))
                ),
            )

        return list(
            await asyncio.gather(
                *(poll_one(case_id, case) for case_id, case in case_by_id.items())
            )
        )

    async def _execute_one(self, case: Case, context: ExecutionContext) -> Rollout:
        case_id = str(case.metadata.get("platform_case_id") or case.input.get("task_id") or "")
        if not case_id:
            raise ValueError(f"case {case.name!r} has no platform_case_id")
        connector_config = _deep_merge(
            self.connector_config,
            {"options": {"repeat": 1}},
        )
        phase = str(case.metadata.get("viking_phase") or "").strip().lower()
        if phase and phase not in {"train", "validation"}:
            raise ValueError(f"unsupported Viking rollout phase: {phase}")
        batch = _rollout_batch_descriptor(case)
        batch_case_ids = batch["case_ids"] if batch is not None else [case_id]
        body = {
            "case_ids": batch_case_ids,
            "workers": min(self.concurrency, len(batch_case_ids)),
            "connector_config": connector_config,
            "runtime_params": dict(self.runtime_params),
            "extra_header": dict(self.extra_header),
        }
        if phase:
            body["phase"] = phase
        if batch is not None and self.batch_coordinator is not None:
            batch_key = _rollout_batch_execution_key(
                platform_task_id=self.platform_task_id,
                batch_id=batch["batch_id"],
                phase=phase,
                case=case,
                context=context,
            )
            idempotency_key = _batch_idempotency_key(
                namespace=self.idempotency_namespace,
                batch_key=batch_key,
                body=body,
            )

            async def submit_batch() -> dict[str, Any]:
                return await self.client.submit_rollout_eval(
                    self.platform_task_id,
                    body=body,
                    idempotency_key=idempotency_key,
                )

            submission = await self.batch_coordinator.get_or_submit(
                batch_key,
                submit_batch,
            )
        else:
            idempotency_key = _idempotency_key(
                namespace=self.idempotency_namespace,
                task_id=self.platform_task_id,
                case=case,
                context=context,
                body=body,
            )
            submission = await self.client.submit_rollout_eval(
                self.platform_task_id,
                body=body,
                idempotency_key=idempotency_key,
            )
        handles = submission["case_rollouts"]
        handle = next(
            (
                item
                for item in handles
                if isinstance(item, dict) and str(item.get("case_id") or "") == case_id
            ),
            None,
        )
        if not isinstance(handle, dict) or not str(handle.get("case_rollout_id") or ""):
            raise PlatformAPIError(f"rollout submission for case {case_id} returned no handle")
        case_rollout_id = str(handle["case_rollout_id"])
        result = await self._poll(case_rollout_id, case=case)
        return _rollout_from_platform_result(
            case=case,
            result=result,
            policy_snapshot_id=context.policy_snapshot_id,
            platform_task_id=self.platform_task_id,
            batch_rollout_id=str(submission.get("batch_rollout_id") or ""),
            case_rollout_id=case_rollout_id,
            require_messages=(
                self.require_messages_for_training and bool(context.metadata.get("training"))
            ),
        )
    async def _poll(self, case_rollout_id: str, *, case: Case) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.timeout_seconds
        transient_attempts = 0
        while True:
            try:
                data = await self.client.get_case_rollout(
                    self.platform_task_id,
                    case_rollout_id,
                )
                transient_attempts = 0
            except (httpx.TransportError, PlatformAPIError) as exc:
                if loop.time() >= deadline or not _is_transient(exc):
                    raise
                transient_attempts += 1
                await asyncio.sleep(min(self.poll_interval_seconds * transient_attempts, 10.0))
                continue

            status = str(data.get("status") or "").strip().lower()
            if status == "completed":
                return data
            if status in _TERMINAL_CASE_STATUSES:
                error = data.get("error") or {"message": f"case rollout status={status}"}
                raise RuntimeError(
                    f"platform rollout {case_rollout_id} failed for case {case.name}: "
                    f"{_safe_json(error)}"
                )
            if loop.time() >= deadline:
                raise TimeoutError(
                    f"platform rollout {case_rollout_id} timed out for case {case.name} "
                    f"after {self.timeout_seconds}s; status={status!r}"
                )
            await asyncio.sleep(self.poll_interval_seconds)


def _rollout_batch_descriptor(case: Case) -> dict[str, Any] | None:
    raw = case.metadata.get(_ROLLOUT_BATCH_METADATA_KEY)
    if not isinstance(raw, dict):
        return None
    batch_id = str(raw.get("batch_id") or "").strip()
    values = raw.get("case_ids")
    if not batch_id or not isinstance(values, list):
        raise ValueError("invalid Ark rollout batch metadata")
    case_ids = list(dict.fromkeys(str(value or "").strip() for value in values))
    if not case_ids or any(not value for value in case_ids):
        raise ValueError("Ark rollout batch contains an empty case id")
    case_id = str(
        case.metadata.get("platform_case_id") or case.input.get("task_id") or ""
    ).strip()
    if case_id not in case_ids:
        raise ValueError(
            f"case {case_id!r} is not part of Ark rollout batch {batch_id!r}"
        )
    return {"batch_id": batch_id, "case_ids": case_ids}


def _rollout_batch_execution_key(
    *,
    platform_task_id: str,
    batch_id: str,
    phase: str,
    case: Case,
    context: ExecutionContext,
) -> str:
    payload = {
        "platform_task_id": platform_task_id,
        "batch_id": batch_id,
        "phase": phase,
        "policy_snapshot_id": context.policy_snapshot_id,
        "execution_metadata": context.metadata,
        "train_trial": case.metadata.get("train_trial"),
        "eval_trial": case.metadata.get("eval_trial"),
    }
    return sha256(_safe_json(payload).encode("utf-8")).hexdigest()


def _batch_idempotency_key(
    *,
    namespace: str,
    batch_key: str,
    body: dict[str, Any],
) -> str:
    digest = sha256(_safe_json(body).encode("utf-8")).hexdigest()[:24]
    return f"{namespace}:batch:{batch_key[:24]}:{digest}"


def _rollout_from_platform_result(
    *,
    case: Case,
    result: dict[str, Any],
    policy_snapshot_id: str,
    platform_task_id: str,
    batch_rollout_id: str,
    case_rollout_id: str,
    require_messages: bool,
) -> Rollout:
    observation = _single_observation(result.get("result"))
    messages = messages_from_platform_result(
        observation,
        case=case,
        require_messages=require_messages,
    )
    evaluation = evaluation_from_platform_result(observation)
    return Rollout(
        case=case,
        messages=messages,
        policy_snapshot_id=policy_snapshot_id,
        evaluation=evaluation,
        metadata={
            "rollout_backend": "ark4_gateway",
            "platform_task_id": platform_task_id,
            "platform_batch_rollout_id": batch_rollout_id,
            "platform_case_rollout_id": case_rollout_id,
            "platform_request_id": result.get("request_id"),
            "platform_result_uri": observation.get("result_uri"),
            "artifacts": list(observation.get("artifacts") or []),
            "trace_status": observation.get("trace_status"),
            "evaluator_status": observation.get("evaluator_status"),
            "completion_seq": result.get("completion_seq"),
        },
    )


def messages_from_platform_result(
    result: dict[str, Any],
    *,
    case: Case,
    require_messages: bool,
) -> list[Message]:
    raw_messages = result.get("messages")
    if not isinstance(raw_messages, list):
        raw_messages = []
    converted: list[Message] = []
    for index, raw in enumerate(raw_messages):
        message = _message_from_payload(raw, index=index)
        if message is not None:
            converted.append(message)

    if require_messages and not converted:
        raise PlatformAPIError(
            f"platform rollout for case {case.name!r} contains no usable messages; "
            "training requires a canonical user/assistant/tool trajectory"
        )

    if not any(message.role == "user" for message in converted):
        query = str(case.input.get("user_query") or "").strip()
        if query:
            converted.insert(0, _text_message("user", query, seed=f"{case.name}:user"))

    final_answer = str(result.get("final_answer") or "").strip()
    if final_answer and not any(
        message.role == "assistant" and message.content.strip() == final_answer
        for message in converted
    ):
        converted.append(_text_message("assistant", final_answer, seed=f"{case.name}:final_answer"))
    if require_messages and not any(message.role == "assistant" for message in converted):
        raise PlatformAPIError(
            f"platform rollout for case {case.name!r} contains no assistant trajectory"
        )
    return converted


def evaluation_from_platform_result(result: dict[str, Any]) -> RubricEvaluation:
    raw = result.get("evaluation")
    if not isinstance(raw, dict):
        evaluator_error = result.get("evaluator_error")
        raise PlatformAPIError(
            "completed platform rollout has no evaluation object"
            + (f": {evaluator_error}" if evaluator_error else "")
        )
    passed = _bool_value(raw.get("passed"), field="evaluation.passed")
    score = _float_score(raw.get("score"), passed=passed)
    criteria = _criterion_results(raw, passed=passed, score=score, result=result)
    return RubricEvaluation(
        passed=passed,
        score=score,
        criterion_results=criteria,
        metadata={
            "platform_evaluation": raw,
            "evaluator_status": result.get("evaluator_status"),
            "evaluator_error": result.get("evaluator_error"),
            "trace_status": result.get("trace_status"),
            "result_uri": result.get("result_uri"),
        },
    )


def _criterion_results(
    raw: dict[str, Any],
    *,
    passed: bool,
    score: float,
    result: dict[str, Any],
) -> list[CriterionResult]:
    raw_criteria = raw.get("criterion_results") or raw.get("criteria")
    criteria: list[CriterionResult] = []
    if isinstance(raw_criteria, list):
        for index, item in enumerate(raw_criteria):
            if not isinstance(item, dict):
                continue
            item_passed = _bool_value(
                item.get("passed"),
                field=f"evaluation.criterion_results[{index}].passed",
            )
            item_score = _float_score(item.get("score"), passed=item_passed)
            criteria.append(
                CriterionResult(
                    criterion_name=str(
                        item.get("criterion_name") or item.get("name") or f"criterion_{index}"
                    ),
                    passed=item_passed,
                    score=item_score,
                    feedback=_string_list(item.get("feedback")),
                    evidence=_string_list(item.get("evidence")),
                    metadata=dict(item.get("metadata") or {}),
                )
            )
    if criteria:
        return criteria

    feedback = _string_list(raw.get("feedback"))
    for key in ("fail_reason", "reason", "message"):
        value = raw.get(key)
        if value is not None and str(value).strip():
            feedback.append(str(value))
    evaluator_error = result.get("evaluator_error")
    if evaluator_error:
        feedback.append(str(evaluator_error))
    evidence = []
    if result.get("result_uri"):
        evidence.append(str(result["result_uri"]))
    return [
        CriterionResult(
            criterion_name="platform_rollout_builtin",
            passed=passed,
            score=score,
            feedback=_dedupe(feedback),
            evidence=evidence,
            metadata={
                key: raw[key]
                for key in ("hard", "soft", "attempt", "evaluator_failure")
                if key in raw
            },
        )
    ]


def _message_from_payload(raw: Any, *, index: int) -> Message | None:
    if not isinstance(raw, dict):
        return _text_message("assistant", _safe_json(raw), seed=f"raw:{index}")
    raw_role = str(raw.get("role") or "").strip().lower()
    if raw_role == "system":
        return None
    if raw_role == "tool":
        tool_id = str(raw.get("tool_call_id") or raw.get("id") or f"tool-{index}")
        tool_name = str(raw.get("name") or raw.get("tool_name") or "tool")
        output = raw.get("content") or raw.get("tool_output") or ""
        return Message(
            id=_message_id(raw, index=index),
            role="assistant",
            parts=[
                ToolPart(
                    tool_id=tool_id,
                    tool_name=tool_name,
                    tool_output=str(output),
                    tool_status="completed",
                )
            ],
        )
    role: Literal["user", "assistant"] = "user" if raw_role == "user" else "assistant"
    parts: list[Any] = []
    raw_parts = raw.get("parts")
    if isinstance(raw_parts, list):
        parts.extend(_parts_from_payload(raw_parts, message_index=index))
    else:
        parts.extend(_content_parts(raw.get("content"), message_index=index))
    parts.extend(_tool_call_parts(raw.get("tool_calls"), message_index=index))
    if not parts:
        fallback = {
            key: value
            for key, value in raw.items()
            if key not in {"id", "role", "created_at", "peer_id"}
        }
        if fallback:
            parts.append(TextPart(text=_safe_json(fallback)))
    if not parts:
        return None
    return Message(
        id=_message_id(raw, index=index),
        role=role,
        parts=parts,
        peer_id=raw.get("peer_id"),
        created_at=raw.get("created_at"),
    )


def _parts_from_payload(raw_parts: list[Any], *, message_index: int) -> list[Any]:
    parts: list[Any] = []
    for part_index, raw in enumerate(raw_parts):
        if not isinstance(raw, dict):
            parts.append(TextPart(text=str(raw)))
            continue
        part_type = str(raw.get("type") or "text")
        if part_type in {"text", "input_text", "output_text"}:
            parts.append(TextPart(text=str(raw.get("text") or raw.get("content") or "")))
        elif part_type == "image_url":
            image_url = raw.get("image_url")
            if isinstance(image_url, dict):
                url = str(image_url.get("url") or "")
                detail = image_url.get("detail")
            else:
                url = str(image_url or raw.get("url") or "")
                detail = raw.get("detail")
            if url:
                parts.append(ImagePart(url=url, detail=detail))
        elif part_type in {"tool", "tool_call", "tool_result"}:
            parts.append(
                ToolPart(
                    tool_id=str(raw.get("tool_id") or raw.get("id") or f"tool-{part_index}"),
                    tool_name=str(raw.get("tool_name") or raw.get("name") or "tool"),
                    tool_input=_mapping_or_none(raw.get("tool_input") or raw.get("arguments")),
                    tool_output=str(raw.get("tool_output") or raw.get("output") or ""),
                    tool_status=str(raw.get("tool_status") or raw.get("status") or "completed"),
                )
            )
        else:
            parts.append(TextPart(text=_safe_json(raw)))
    return parts


def _content_parts(content: Any, *, message_index: int) -> list[Any]:
    if content is None:
        return []
    if isinstance(content, str):
        return [TextPart(text=content)] if content else []
    if isinstance(content, list):
        return _parts_from_payload(content, message_index=message_index)
    return [TextPart(text=_safe_json(content))]


def _tool_call_parts(raw_calls: Any, *, message_index: int) -> list[ToolPart]:
    if not isinstance(raw_calls, list):
        return []
    parts: list[ToolPart] = []
    for call_index, raw in enumerate(raw_calls):
        if not isinstance(raw, dict):
            continue
        function = raw.get("function") if isinstance(raw.get("function"), dict) else raw
        parts.append(
            ToolPart(
                tool_id=str(raw.get("id") or f"tool-{message_index}-{call_index}"),
                tool_name=str(function.get("name") or raw.get("name") or "tool"),
                tool_input=_mapping_or_none(function.get("arguments")),
                tool_output=str(raw.get("output") or ""),
                tool_status="completed" if raw.get("output") is not None else "pending",
            )
        )
    return parts


def _single_observation(raw_result: Any) -> dict[str, Any]:
    if not isinstance(raw_result, dict):
        raise PlatformAPIError("completed platform rollout has no result object")
    observations = raw_result.get("observations")
    if observations is None:
        return raw_result
    if not isinstance(observations, list) or len(observations) != 1:
        raise PlatformAPIError(
            "Ark adapter submits repeat=1 and expects exactly one observation per case rollout"
        )
    observation = observations[0]
    if not isinstance(observation, dict):
        raise PlatformAPIError("platform observation must be an object")
    return observation


def _idempotency_key(
    *,
    namespace: str,
    task_id: str,
    case: Case,
    context: ExecutionContext,
    body: dict[str, Any],
) -> str:
    payload = {
        "task_id": task_id,
        "case_name": case.name,
        "case_signature": case.task_signature,
        "case_id": case.metadata.get("platform_case_id"),
        "policy_snapshot_id": context.policy_snapshot_id,
        "execution_metadata": context.metadata,
        "body": body,
    }
    stable = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = sha256(stable.encode("utf-8")).hexdigest()[:32]
    safe_namespace = "".join(ch if ch.isalnum() or ch in "_.:-" else "-" for ch in namespace)
    return f"{safe_namespace}:{digest}"[:200]


def _message_id(raw: dict[str, Any], *, index: int) -> str:
    value = raw.get("id")
    if value is not None and str(value).strip():
        return str(value)
    stable = json.dumps(raw, ensure_ascii=False, sort_keys=True, default=str)
    return f"ark4-message-{index}-{sha256(stable.encode('utf-8')).hexdigest()[:16]}"


def _text_message(
    role: Literal["user", "assistant"],
    text: str,
    *,
    seed: str,
) -> Message:
    digest = sha256(f"{role}:{seed}:{text}".encode("utf-8")).hexdigest()[:16]
    return Message(
        id=f"ark4-{role}-{digest}",
        role=role,
        parts=[TextPart(text=text)],
    )


def _mapping_or_none(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {"raw": value} if value else None
        return dict(parsed) if isinstance(parsed, dict) else {"value": parsed}
    if value is None:
        return None
    return {"value": value}


def _float_score(value: Any, *, passed: bool) -> float:
    if value is None:
        return 1.0 if passed else 0.0
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise PlatformAPIError(f"invalid platform evaluation score: {value!r}") from exc


def _bool_value(value: Any, *, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1"}:
            return True
        if normalized in {"false", "0"}:
            return False
    if value is None:
        return False
    raise PlatformAPIError(f"invalid platform boolean {field}: {value!r}")


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list | tuple):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)] if str(value).strip() else []


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TransportError):
        return True
    return isinstance(exc, PlatformAPIError) and exc.status_code in {
        408,
        425,
        429,
        500,
        502,
        503,
        504,
    }


def _safe_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        return str(value)
