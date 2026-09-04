from __future__ import annotations

import httpx
import pytest
from platform_client import PlatformClientConfig, TrainingPlatformClient


@pytest.mark.asyncio
async def test_platform_client_runs_task_rollout_and_completion_flow() -> None:
    task_reads = 0

    def gateway_handler(request: httpx.Request) -> httpx.Response:
        nonlocal task_reads
        assert request.headers["x-vaka-request-source"] == "ark-lx"
        if request.method == "GET" and request.url.path == "/inspect/resources/resources":
            assert request.url.params["applicable_agent_id"] == "ark"
            return httpx.Response(
                200,
                json={
                    "ts": 0,
                    "data": {
                        "resources": [
                            {
                                "resource_id": "rm-lane-evolving",
                                "resource_type": "rollout_concurrency",
                                "scope": "lane",
                                "agent_id": "ark",
                                "lane_key": "evolving",
                                "enabled": True,
                            }
                        ]
                    },
                },
            )
        if request.method == "POST" and request.url.path == "/inspect/training/tasks":
            return httpx.Response(
                200,
                json={"ts": 1, "data": {"task_id": "task-1", "status": "pending"}},
            )
        if request.method == "GET" and request.url.path == "/inspect/training/tasks/task-1":
            task_reads += 1
            if task_reads == 1:
                return httpx.Response(
                    200,
                    json={"ts": 2, "data": {"task_id": "task-1", "status": "pending"}},
                )
            return httpx.Response(
                200,
                json={
                    "ts": 3,
                    "data": {
                        "task_id": "task-1",
                        "status": "running",
                        "current_step": "OV_WAIT",
                        "steps": [{"step": "OV_WAIT", "status": "running"}],
                    },
                },
            )
        if request.method == "GET" and request.url.path == "/inspect/casehub/cases":
            assert request.url.params["caseset_id"] == "cs-1"
            return httpx.Response(200, json={"cases": [{"case_id": "case-1"}]})
        if request.method == "GET" and request.url.path == "/inspect/casehub/cases/case-1":
            return httpx.Response(
                200,
                json={"case_id": "case-1", "envelope": {"input_prompt": "hello"}},
            )
        if request.method == "GET" and request.url.path.endswith("/rollout-source-cases"):
            assert request.url.params["phase"] == "train"
            return httpx.Response(
                200,
                json={
                    "ts": 3,
                    "data": {
                        "phase": "train",
                        "total": 1,
                        "cases": [{"case_id": "101", "input": {"prompt": "hello"}}],
                    },
                },
            )
        if request.method == "POST" and request.url.path.endswith("/rollout-eval"):
            assert request.headers["Idempotency-Key"] == "idem-1"
            return httpx.Response(
                202,
                json={
                    "ts": 4,
                    "data": {
                        "batch_rollout_id": "batch-1",
                        "case_rollouts": [{"case_id": "case-1", "case_rollout_id": "rollout-1"}],
                    },
                },
            )
        if request.method == "POST" and request.url.path.endswith(
            "/signals/external-training-completed"
        ):
            return httpx.Response(
                200,
                json={
                    "ts": 5,
                    "data": {
                        "task_id": "task-1",
                        "status": "succeeded",
                        "signal": "external-training-completed",
                    },
                },
            )
        if (
            request.method == "GET"
            and request.url.path == "/inspect/training/tasks/task-1/rollout-eval/rollout-1"
        ):
            return httpx.Response(
                200,
                json={
                    "ts": 6,
                    "data": {
                        "case_rollout_id": "rollout-1",
                        "case_id": "case-1",
                        "status": "completed",
                        "result": {"evaluation": {"passed": True, "score": 1}},
                    },
                },
            )
        raise AssertionError(f"unexpected gateway request: {request.method} {request.url}")

    client = TrainingPlatformClient(
        PlatformClientConfig(
            gateway_base_url="https://gateway.test",
            api_key="secret",
            project_id="project-1",
            vaka_request_source="ark-lx",
            headers={"X-Vaka-Request-Source": "must-be-overridden"},
        ),
        transport=httpx.MockTransport(gateway_handler),
    )
    try:
        lane = await client.resolve_rollout_lane_resource(
            agent_id="ark",
            lane_key="evolving",
        )
        assert lane["resource_id"] == "rm-lane-evolving"
        created = await client.create_training_task({"workflow_id": "ov_external_training"})
        assert created["task_id"] == "task-1"
        ready = await client.wait_for_ov_wait(
            "task-1",
            poll_interval_seconds=0.001,
            timeout_seconds=1,
        )
        assert ready["current_step"] == "OV_WAIT"
        assert await client.list_cases(caseset_id="cs-1", dataset_id=None, limit=10, offset=0)
        assert await client.get_case("case-1")
        source = await client.list_rollout_source_cases(
            "task-1", phase="train", page=1, page_size=100
        )
        assert source["cases"][0]["case_id"] == "101"
        submission = await client.submit_rollout_eval(
            "task-1",
            body={"case_ids": ["case-1"]},
            idempotency_key="idem-1",
        )
        assert submission["batch_rollout_id"] == "batch-1"
        rollout = await client.get_case_rollout("task-1", "rollout-1")
        assert rollout["status"] == "completed"
        completion = await client.complete_external_training(
            "task-1",
            idempotency_key="complete-1",
        )
        assert completion["status"] == "succeeded"
    finally:
        await client.close()
