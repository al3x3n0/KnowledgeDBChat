import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.endpoints import scientific_sandbox_profiles
from app.core.database import get_db


@pytest.fixture
def sandbox_profile_client(db_session, admin_user):
    app = FastAPI()
    app.include_router(
        scientific_sandbox_profiles.router,
        prefix="/api/v1/scientific-sandbox-profiles",
    )

    def override_get_db():
        return db_session

    async def override_admin_user():
        return admin_user

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[
        scientific_sandbox_profiles.get_current_active_user
    ] = override_admin_user
    app.dependency_overrides[
        scientific_sandbox_profiles.require_admin
    ] = override_admin_user

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


def test_list_scientific_sandbox_profiles_seeds_builtin_profiles(
    sandbox_profile_client,
):
    response = sandbox_profile_client.get("/api/v1/scientific-sandbox-profiles")

    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] >= 3
    items = {item["id"]: item for item in payload["items"]}
    assert "scientific-compiler-sandbox" in items
    assert "scientific-microarchitecture-sandbox" in items
    assert "scientific-generic-sandbox" in items
    assert items["scientific-compiler-sandbox"]["system_managed"] is True
    assert items["scientific-compiler-sandbox"]["track_type"] == "compiler"


def test_create_update_and_delete_custom_scientific_sandbox_profile(
    sandbox_profile_client,
):
    create_response = sandbox_profile_client.post(
        "/api/v1/scientific-sandbox-profiles",
        json={
            "id": "custom-generic-sandbox",
            "name": "Custom Generic Sandbox",
            "description": "Custom validation profile for bounded generic validation.",
            "track_type": "generic",
            "backend": "docker",
            "docker_image": "python:3.11-slim",
            "timeout_seconds": 600,
            "resource_caps": {"memory_mb": 1024, "cpus": 1.0, "pids_limit": 128},
            "allowed_benchmark_families": ["generic_validation"],
            "allowed_perf_collectors": ["benchmark_output"],
            "required_capabilities": ["repo_reconstruction"],
            "toolchains": ["python", "pytest"],
            "budget_limit_default": 15.5,
            "enabled": True,
            "is_default": False,
        },
    )

    assert create_response.status_code == 201
    created = create_response.json()
    assert created["id"] == "custom-generic-sandbox"
    assert created["system_managed"] is False
    assert created["created_by_user_id"] is not None

    update_response = sandbox_profile_client.patch(
        "/api/v1/scientific-sandbox-profiles/custom-generic-sandbox",
        json={
            "name": "Custom Generic Sandbox v2",
            "enabled": False,
            "budget_limit_default": 18.0,
        },
    )

    assert update_response.status_code == 200
    updated = update_response.json()
    assert updated["name"] == "Custom Generic Sandbox v2"
    assert updated["enabled"] is False
    assert updated["budget_limit_default"] == 18.0

    delete_response = sandbox_profile_client.delete(
        "/api/v1/scientific-sandbox-profiles/custom-generic-sandbox"
    )
    assert delete_response.status_code == 204

    get_response = sandbox_profile_client.get(
        "/api/v1/scientific-sandbox-profiles/custom-generic-sandbox"
    )
    assert get_response.status_code == 404


def test_create_scientific_sandbox_profile_rejects_disallowed_docker_image(
    sandbox_profile_client,
):
    response = sandbox_profile_client.post(
        "/api/v1/scientific-sandbox-profiles",
        json={
            "id": "blocked-sandbox",
            "name": "Blocked Sandbox",
            "track_type": "generic",
            "backend": "docker",
            "docker_image": "alpine:latest",
            "timeout_seconds": 300,
            "resource_caps": {"memory_mb": 512, "cpus": 1.0, "pids_limit": 128},
            "allowed_benchmark_families": ["generic_validation"],
            "allowed_perf_collectors": ["benchmark_output"],
            "required_capabilities": ["repo_reconstruction"],
            "toolchains": ["python"],
            "budget_limit_default": 10.0,
        },
    )

    assert response.status_code == 400
    assert "allowlist" in response.json()["detail"]
