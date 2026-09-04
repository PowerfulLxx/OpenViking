from __future__ import annotations

from typing import Any

import pytest
from case_loader import ArkCaseRepository, VikingCaseRepository


class FakeCaseClient:
    def __init__(self) -> None:
        self.rows = [
            {"case_id": "case-1", "name": "train case"},
            {"case_id": "case-2", "name": "test case"},
            {"case_id": "case-3", "name": "second train case"},
        ]
        self.details = {
            "case-1": {
                "case_id": "case-1",
                "dataset_id": "dataset-1",
                "envelope": {
                    "input_prompt": "train prompt",
                    "expected_answer": "train answer",
                    "metadata": {"split": "train"},
                },
            },
            "case-2": {
                "case_id": "case-2",
                "dataset_id": "dataset-1",
                "envelope": {
                    "input_prompt": "test prompt",
                    "expected_answer": "test answer",
                    "metadata": {"split": "test"},
                },
            },
            "case-3": {
                "case_id": "case-3",
                "dataset_id": "dataset-1",
                "envelope": {
                    "fields": {"prompt": "nested train prompt"},
                    "metadata": {"split": "training"},
                },
            },
        }

    async def list_cases(
        self,
        *,
        caseset_id: str | None,
        dataset_id: str | None,
        limit: int,
        offset: int,
    ) -> Any:
        assert caseset_id is None
        assert dataset_id == "dataset-1"
        return {"items": self.rows[offset : offset + limit]}

    async def get_case(self, case_id: str) -> Any:
        return {"data": self.details[case_id]}


@pytest.mark.asyncio
async def test_repository_maps_casehub_envelopes_and_splits() -> None:
    repository = ArkCaseRepository(
        client=FakeCaseClient(),  # type: ignore[arg-type]
        dataset_ids=["dataset-1"],
        split_field="metadata.split",
        page_size=2,
    )

    train_cases = await repository.cases_for_split("train")
    test_cases = await repository.cases_for_split("test")

    assert [case.metadata["platform_case_id"] for case in train_cases] == [
        "case-1",
        "case-3",
    ]
    assert train_cases[0].input["user_query"] == "train prompt"
    assert train_cases[0].input["expected_answer"] == "train answer"
    assert train_cases[1].input["user_query"] == "nested train prompt"
    assert [case.metadata["platform_case_id"] for case in test_cases] == ["case-2"]


@pytest.mark.asyncio
async def test_repository_can_load_an_explicit_case_id_without_listing() -> None:
    repository = ArkCaseRepository(
        client=FakeCaseClient(),  # type: ignore[arg-type]
        dataset_ids=["dataset-1"],
        case_ids=["case-2"],
        split_field="metadata.split",
    )

    cases = await repository.cases_for_split("test")

    assert [case.metadata["platform_case_id"] for case in cases] == ["case-2"]


class FakeVikingClient:
    async def list_rollout_source_cases(
        self,
        task_id: str,
        *,
        phase: str,
        page: int,
        page_size: int,
    ) -> dict[str, Any]:
        assert task_id == "task-viking"
        assert page_size == 500
        rows = {
            "train": [
                {
                    "case_id": "101",
                    "input": {"prompt": "train prompt"},
                    "expected_answer": "train answer",
                    "rubric": {
                        "name": "viking",
                        "criteria": [{"name": "answer_score", "weight": 1.0}],
                    },
                    "metadata": {"viking_row_id": 101},
                }
            ],
            "validation": [
                {"case_id": "201", "input": {"prompt": "validation prompt"}}
            ],
        }[phase]
        return {"phase": phase, "total": len(rows), "cases": rows if page == 1 else []}


@pytest.mark.asyncio
async def test_viking_repository_maps_phase_source_cases() -> None:
    repository = VikingCaseRepository(
        client=FakeVikingClient(),  # type: ignore[arg-type]
        task_id="task-viking",
        case_ids=["101"],
    )

    train_cases = await repository.cases_for_split("train")
    validation_cases = await repository.cases_for_split("dev")

    assert [case.metadata["platform_case_id"] for case in train_cases] == ["101"]
    assert train_cases[0].metadata["viking_phase"] == "train"
    assert train_cases[0].input["expected_answer"] == "train answer"
    assert validation_cases == []
