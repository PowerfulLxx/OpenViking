from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from start_adapter import SCRIPT_DIR, load_config, parse_args


def write_config(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "service": {"port": 1944, "admin_token": "secret"},
                "platform": {
                    "gateway_base_url": "https://gateway.test",
                    "api_key": "platform-secret",
                    "project_id": "00000000000000000000000000000001",
                    "vaka_request_source": "ark-lx",
                },
                "training_task": {
                    "agent_lane_key": "evolving",
                    "agent_execution": {
                        "contract_id": "ark.viking-rollout",
                        "contract_version": "3",
                        "schema_digest": "sha256:test",
                        "values": {
                            "memory_openviking_target": "ov-ark-test",
                        },
                    },
                },
                "rollout": {"require_messages_for_training": True},
            }
        ),
        encoding="utf-8",
    )


def test_load_config_uses_only_file(tmp_path: Path) -> None:
    path = tmp_path / "adapter.local.json"
    write_config(path)

    config = load_config(path)

    assert config.platform.gateway_base_url == "https://gateway.test"
    assert config.platform.api_key == "platform-secret"
    assert config.platform.vaka_request_source == "ark-lx"
    assert config.training_task.agent_lane_key == "evolving"
    assert config.training_task.agent_execution["contract_id"] == "ark.viking-rollout"
    assert config.rollout.extra_header == {"x-vaka-request-source": "ark-lx"}
    assert config.service.dataset == "ark4-0"


def test_config_argument_defaults_next_to_start_script(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["start_adapter.py"])

    args = parse_args()

    assert Path(args.config) == SCRIPT_DIR / "adapter_config.local.json"


def test_load_config_rejects_protected_rollout_header(tmp_path: Path) -> None:
    path = tmp_path / "adapter.local.json"
    write_config(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["rollout"]["extra_header"] = {"x-tt-backend": "evolution"}
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="protected header"):
        load_config(path)


def test_load_config_rejects_mismatched_rollout_vaka_source(tmp_path: Path) -> None:
    path = tmp_path / "adapter.local.json"
    write_config(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["rollout"]["extra_header"] = {
        "x-vaka-request-source": "vaka-agentmemory"
    }
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="must match platform.vaka_request_source"):
        load_config(path)


def test_load_config_rejects_project_display_name(tmp_path: Path) -> None:
    path = tmp_path / "adapter.local.json"
    write_config(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["platform"]["project_id"] = "ov-ark-test"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="not the project display name"):
        load_config(path)


def test_load_config_resolves_memory_key_and_matching_target(tmp_path: Path) -> None:
    path = tmp_path / "adapter.local.json"
    write_config(path)
    ov_config = tmp_path / "openviking.conf"
    ov_config.write_text(
        json.dumps({"server": {"root_api_key": "local-root-key"}}),
        encoding="utf-8",
    )
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["rollout"]["runtime_params"] = {
        "memory": {
            "enabled": True,
            "mode": "read_only",
            "openviking_target": "ov-ark-test",
        }
    }
    raw["memory_proxy"] = {
        "enabled": True,
        "openviking_config_file": "openviking.conf",
        "openviking_api_key_json_path": "server.root_api_key",
        "openviking_account_id": "local-account",
        "openviking_user_id": "local-user",
    }
    raw["training_task"]["agent_execution"]["values"][
        "memory_openviking_target"
    ] = "ov-ark-test"
    path.write_text(json.dumps(raw), encoding="utf-8")

    config = load_config(path)

    assert config.memory_proxy.enabled is True
    assert config.memory_proxy.openviking_target == "ov-ark-test"
    assert config.memory_proxy.openviking_api_key == "local-root-key"
    assert config.memory_proxy.openviking_account_id == "local-account"
    assert config.memory_proxy.openviking_user_id == "local-user"
    assert config.memory_proxy.event_log_file == tmp_path / "memory_proxy_events.local.jsonl"


def test_load_config_rejects_mismatched_memory_target(tmp_path: Path) -> None:
    path = tmp_path / "adapter.local.json"
    write_config(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["rollout"]["runtime_params"] = {
        "memory": {"enabled": True, "openviking_target": "target-a"}
    }
    raw["memory_proxy"] = {
        "enabled": True,
        "openviking_target": "target-b",
        "openviking_api_key": "local-user-key",
    }
    raw["training_task"]["agent_execution"]["values"][
        "memory_openviking_target"
    ] = "target-b"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="must equal"):
        load_config(path)


def test_load_config_rejects_mismatched_task_memory_target(tmp_path: Path) -> None:
    path = tmp_path / "adapter.local.json"
    write_config(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["rollout"]["runtime_params"] = {
        "memory": {"enabled": True, "openviking_target": "target-a"}
    }
    raw["memory_proxy"] = {
        "enabled": True,
        "openviking_target": "target-a",
        "openviking_api_key": "local-user-key",
    }
    raw["training_task"]["agent_execution"]["values"][
        "memory_openviking_target"
    ] = "target-b"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="training_task.agent_execution"):
        load_config(path)
