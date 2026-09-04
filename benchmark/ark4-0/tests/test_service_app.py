from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest
from service_app import ArkAdapterServiceConfig, create_app

from openviking.session.train import Case, ExperienceSet, Rubric, RubricCriterion
from openviking.session.train.components.dataset_service import case_to_dict, policy_set_to_dict


def make_case() -> Case:
    return Case(
        name="adapter case",
        task_signature="adapter-case-signature",
        input={"task_id": "case-1", "user_query": "hello"},
        rubric=Rubric(
            name="platform",
            description="",
            criteria=[
                RubricCriterion(
                    name="platform",
                    description="",
                    required=True,
                    weight=1.0,
                )
            ],
        ),
        metadata={"platform_case_id": "case-1"},
    )


class CompletedPlatformClient:
    def __init__(self) -> None:
        self.completed = False
        self.created_count = 0
        self.created_bodies: list[dict[str, Any]] = []

    async def create_training_task(self, body: dict[str, Any]) -> dict[str, Any]:
        self.created_bodies.append(body)
        self.created_count += 1
        return {"task_id": f"task-{self.created_count}", "status": "pending"}

    async def get_case(self, case_id: str) -> dict[str, Any]:
        dataset_id = {"case-1": "dataset-1", "case-2": "dataset-2"}[case_id]
        return {
            "case_id": case_id,
            "dataset_id": dataset_id,
            "name": f"adapter {case_id}",
            "envelope": {"user_query": f"hello {case_id}"},
        }

    async def list_rollout_source_cases(
        self,
        task_id: str,
        *,
        phase: str,
        page: int,
        page_size: int,
    ) -> dict[str, Any]:
        assert task_id.startswith("task-")
        rows = [
            {
                "case_id": "101",
                "input": {"prompt": "viking train prompt"},
                "expected_answer": "viking train answer",
                "metadata": {"viking_row_id": 101},
            }
        ] if phase == "train" else []
        return {"phase": phase, "page": page, "page_size": page_size, "total": len(rows), "cases": rows}

    async def wait_for_ov_wait(
        self,
        task_id: str,
        *,
        poll_interval_seconds: float,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        assert poll_interval_seconds > 0
        assert timeout_seconds > 0
        return {"task_id": task_id, "status": "running", "current_step": "OV_WAIT"}

    async def get_training_task(self, task_id: str) -> dict[str, Any]:
        assert task_id in {"task-1", "task-existing"}
        return {"task_id": task_id, "status": "running", "current_step": "OV_WAIT"}

    async def complete_external_training(
        self,
        task_id: str,
        *,
        idempotency_key: str,
    ) -> dict[str, Any]:
        assert task_id == "task-1"
        assert idempotency_key.endswith(":task-1:complete")
        self.completed = True
        return {"task_id": task_id, "status": "succeeded"}

    async def submit_rollout_eval(
        self,
        task_id: str,
        *,
        body: dict[str, Any],
        idempotency_key: str,
    ) -> dict[str, Any]:
        assert task_id == "task-1"
        assert body["case_ids"] == ["case-1"]
        assert idempotency_key
        return {
            "batch_rollout_id": "batch-1",
            "case_rollouts": [{"case_id": "case-1", "case_rollout_id": "case-rollout-1"}],
        }

    async def get_case_rollout(self, task_id: str, case_rollout_id: str) -> dict[str, Any]:
        return {
            "status": "completed",
            "result": {
                "final_answer": "done",
                "messages": [
                    {"id": "m1", "role": "user", "content": "hello"},
                    {"id": "m2", "role": "assistant", "content": "done"},
                ],
                "evaluation": {"passed": True, "score": 1.0},
                "evaluator_status": "succeeded",
            },
        }


@pytest.mark.asyncio
async def test_generic_service_contract_executes_platform_rollout() -> None:
    platform_client = CompletedPlatformClient()
    app = create_app(
        client=platform_client,  # type: ignore[arg-type]
        config=ArkAdapterServiceConfig(
            dataset="ark4-0",
            domain="ark",
            rollout_concurrency=2,
            rollout_poll_interval_seconds=0.001,
            rollout_timeout_seconds=1,
            admin_token="admin-secret",
        ),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://adapter.test",
    ) as client:
        start_response = await client.post(
            "/v1/runs/start",
            json={
                "run_id": "run-1",
                "dataset": "ark4-0",
                "domain": "ark",
                "concurrency": 30,
                "casehub": {"dataset_ids": ["dataset-1"], "case_ids": ["case-1"]},
            },
        )
        assert start_response.status_code == 200
        assert start_response.json()["task_id"] == "task-1"
        assert platform_client.created_count == 1
        assert platform_client.created_bodies[0]["casehub_dataset_ids"] == ["dataset-1"]
        assert platform_client.created_bodies[0]["workers"] == 30
        assert start_response.json()["concurrency"] == 30
        assert start_response.json()["task_casehub_dataset_ids"] == ["dataset-1"]
        assert platform_client.created_bodies[0]["task_name"].endswith("_run-1")

        duplicate_start = await client.post(
            "/v1/runs/start",
            json={
                "run_id": "run-1",
                "dataset": "ark4-0",
                "domain": "ark",
                "concurrency": 30,
                "casehub": {"dataset_ids": ["dataset-1"], "case_ids": ["case-1"]},
            },
        )
        assert duplicate_start.json()["task_id"] == "task-1"
        assert platform_client.created_count == 1

        case_response = await client.post(
            "/v1/cases/query",
            json={
                "dataset": "ark4-0",
                "domain": "ark",
                "split": "train",
                "limit": 10,
                "filters": {"_openviking_benchmark_run_id": "run-1"},
            },
        )
        assert case_response.status_code == 200
        assert len(case_response.json()["cases"]) == 1
        assert case_response.json()["cases"][0]["metadata"][
            "_ark_rollout_batch"
        ]["case_ids"] == ["case-1"]

        execute_response = await client.post(
            "/v1/rollouts/execute",
            json={
                "case": case_to_dict(make_case()),
                "policy_set": policy_set_to_dict(
                    ExperienceSet(
                        root_uri="viking://user/memories/experiences",
                        policies=[],
                    )
                ),
                "execution_context": {
                    "policy_snapshot_id": "snapshot-1",
                    "metadata": {"training": True, "epoch": 0},
                },
                "options": {"_openviking_benchmark_run_id": "run-1"},
            },
        )
        assert execute_response.status_code == 200
        execution_id = execute_response.json()["execution_id"]

        for _ in range(100):
            poll_response = await client.get(f"/v1/rollouts/executions/{execution_id}")
            payload = poll_response.json()
            if payload["status"] == "completed":
                break
            await asyncio.sleep(0.001)
        else:
            raise AssertionError("generic rollout execution did not complete")

        assert payload["rollout"]["evaluation"]["passed"] is True
        assert [message["role"] for message in payload["rollout"]["messages"]] == [
            "user",
            "assistant",
        ]

        unauthorized = await client.get("/admin/platform-runs")
        assert unauthorized.status_code == 401

        admin_headers = {"X-Ark4-Admin-Token": "admin-secret"}
        task_response = await client.get("/admin/platform-runs/run-1", headers=admin_headers)
        assert task_response.status_code == 200
        assert task_response.json()["task_id"] == "task-1"

        complete_response = await client.post(
            "/v1/runs/run-1/complete",
        )
        assert complete_response.status_code == 200
        assert complete_response.json()["completion"]["status"] == "succeeded"
        assert platform_client.completed is True


@pytest.mark.asyncio
async def test_each_run_has_its_own_casehub_selection() -> None:
    platform_client = CompletedPlatformClient()
    app = create_app(
        client=platform_client,  # type: ignore[arg-type]
        config=ArkAdapterServiceConfig(dataset="ark4-0", domain="ark"),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://adapter.test",
    ) as client:
        for run_id, dataset_id, case_id in (
            ("run-1", "dataset-1", "case-1"),
            ("run-2", "dataset-2", "case-2"),
        ):
            response = await client.post(
                "/v1/runs/start",
                json={
                    "run_id": run_id,
                    "dataset": "ark4-0",
                    "domain": "ark",
                    "casehub": {"dataset_ids": [dataset_id], "case_ids": [case_id]},
                },
            )
            assert response.status_code == 200
            assert response.json()["casehub_case_ids"] == [case_id]
            assert response.json()["case_count"] == 1

            cases = await client.post(
                "/v1/cases/query",
                json={
                    "dataset": "ark4-0",
                    "domain": "ark",
                    "split": "train",
                    "limit": 10,
                    "filters": {"_openviking_benchmark_run_id": run_id},
                },
            )
            assert cases.status_code == 200
            assert [item["input"]["task_id"] for item in cases.json()["cases"]] == [case_id]

    assert [body["casehub_dataset_ids"] for body in platform_client.created_bodies] == [
        ["dataset-1"],
        ["dataset-2"],
    ]


@pytest.mark.asyncio
async def test_viking_external_run_uses_v2_task_body_and_phase_source() -> None:
    platform_client = CompletedPlatformClient()
    app = create_app(
        client=platform_client,  # type: ignore[arg-type]
        config=ArkAdapterServiceConfig(
            dataset="ark4-0",
            domain="ark",
            workflow_id="ark_viking_external_training",
            task_body={
                "schema_version": "training-task-request.v2",
                "name": "viking-external",
                "experiment_id": "exp-external",
                "viking_experiment_sets": [
                    {"experiment_set_id": 371, "version": "V1", "role": "train"},
                    {"experiment_set_id": 372, "version": "V1", "role": "eval"},
                ],
            },
        ),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://adapter.test",
    ) as client:
        response = await client.post(
            "/v1/runs/start",
            json={
                "run_id": "viking-run",
                "dataset": "ark4-0",
                "domain": "ark",
                "concurrency": 1,
                "casehub": {"dataset_ids": ["viking"], "case_ids": ["101"]},
            },
        )

        assert response.status_code == 200
        assert response.json()["case_count"] == 1
        assert platform_client.created_bodies[0]["name"] == "viking-external_viking-run"
        assert "casehub_dataset_ids" not in platform_client.created_bodies[0]


@pytest.mark.asyncio
async def test_viking_external_run_can_attach_existing_task() -> None:
    platform_client = CompletedPlatformClient()
    app = create_app(
        client=platform_client,  # type: ignore[arg-type]
        config=ArkAdapterServiceConfig(
            dataset="ark4-0",
            domain="ark",
            workflow_id="ark_viking_external_training",
            existing_task_id="task-existing",
        ),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://adapter.test",
    ) as client:
        response = await client.post(
            "/v1/runs/start",
            json={
                "run_id": "attached-run",
                "dataset": "ark4-0",
                "domain": "ark",
                "concurrency": 1,
            },
        )

    assert response.status_code == 200
    assert response.json()["task_id"] == "task-existing"
    assert platform_client.created_bodies == []


@pytest.mark.asyncio
async def test_case_query_records_exact_requested_batch_page() -> None:
    class ThreeCasePlatformClient(CompletedPlatformClient):
        async def list_rollout_source_cases(
            self,
            task_id: str,
            *,
            phase: str,
            page: int,
            page_size: int,
        ) -> dict[str, Any]:
            rows = [
                {
                    "case_id": str(row_id),
                    "input": {"prompt": f"prompt {row_id}"},
                    "expected_answer": f"answer {row_id}",
                    "metadata": {"viking_row_id": row_id},
                }
                for row_id in (101, 102, 103)
            ] if phase == "train" else []
            start = max(0, page - 1) * page_size
            selected = rows[start : start + page_size]
            return {
                "phase": phase,
                "page": page,
                "page_size": page_size,
                "total": len(rows),
                "cases": selected,
            }

    platform_client = ThreeCasePlatformClient()
    app = create_app(
        client=platform_client,  # type: ignore[arg-type]
        config=ArkAdapterServiceConfig(
            dataset="ark4-0",
            domain="ark",
            workflow_id="ark_viking_external_training",
            existing_task_id="task-existing",
        ),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://adapter.test",
    ) as client:
        started = await client.post(
            "/v1/runs/start",
            json={
                "run_id": "attached-run",
                "dataset": "ark4-0",
                "domain": "ark",
                "concurrency": 2,
            },
        )
        assert started.status_code == 200
        response = await client.post(
            "/v1/cases/query",
            json={
                "dataset": "ark4-0",
                "domain": "ark",
                "split": "train",
                "limit": 2,
                "filters": {"_openviking_benchmark_run_id": "attached-run"},
            },
        )

    assert response.status_code == 200
    cases = response.json()["cases"]
    assert len(cases) == 2
    descriptors = [
        case["metadata"]["_ark_rollout_batch"] for case in cases
    ]
    assert descriptors[0] == descriptors[1]
    assert descriptors[0]["case_ids"] == ["101", "102"]


@pytest.mark.asyncio
async def test_casehub_selection_is_validated_before_task_creation() -> None:
    platform_client = CompletedPlatformClient()
    app = create_app(
        client=platform_client,  # type: ignore[arg-type]
        config=ArkAdapterServiceConfig(dataset="ark4-0", domain="ark"),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://adapter.test",
    ) as client:
        response = await client.post(
            "/v1/runs/start",
            json={
                "run_id": "run-invalid",
                "dataset": "ark4-0",
                "domain": "ark",
                "casehub": {"dataset_ids": ["dataset-1"], "case_ids": ["case-2"]},
            },
        )

    assert response.status_code == 400
    assert "do not belong" in response.json()["detail"]
    assert platform_client.created_count == 0


@pytest.mark.asyncio
async def test_case_query_can_select_one_dataset_from_multi_dataset_run() -> None:
    platform_client = CompletedPlatformClient()
    app = create_app(
        client=platform_client,  # type: ignore[arg-type]
        config=ArkAdapterServiceConfig(dataset="ark4-0", domain="ark"),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://adapter.test",
    ) as client:
        start_response = await client.post(
            "/v1/runs/start",
            json={
                "run_id": "run-multi",
                "dataset": "ark4-0",
                "domain": "ark",
                "casehub": {
                    "dataset_ids": ["dataset-1", "dataset-2"],
                    "case_ids": ["case-1", "case-2"],
                    "task_dataset_ids": ["dataset-1"],
                },
            },
        )
        assert start_response.status_code == 200
        assert platform_client.created_bodies[0]["casehub_dataset_ids"] == ["dataset-1"]
        assert start_response.json()["task_casehub_dataset_ids"] == ["dataset-1"]

        cases = await client.post(
            "/v1/cases/query",
            json={
                "dataset": "ark4-0",
                "domain": "ark",
                "split": "train",
                "limit": 10,
                "filters": {
                    "_openviking_benchmark_run_id": "run-multi",
                    "_openviking_casehub_dataset_ids": ["dataset-2"],
                },
            },
        )

    assert cases.status_code == 200
    assert [item["input"]["task_id"] for item in cases.json()["cases"]] == ["case-2"]
