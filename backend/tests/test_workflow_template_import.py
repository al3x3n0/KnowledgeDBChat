"""Importing a workflow template answers with the workflow it created.

The handler built its response by hand and left out `user_id` and the ids of
every node and edge, all of which the response schema requires. So the import
committed the workflow and then answered 500 -- and a retry was refused,
because a workflow of that name now existed.
"""

import pytest

from app.services.workflow_templates import WORKFLOW_TEMPLATES

pytestmark = pytest.mark.unit


def test_importing_a_template_returns_the_new_workflow(client, auth_headers, test_user):
    template = WORKFLOW_TEMPLATES[0]

    response = client.post(
        f"/api/v1/workflows/templates/{template['template_id']}/import", headers=auth_headers
    )

    assert response.status_code == 200, response.text
    workflow = response.json()["workflow"]
    assert workflow["name"] == template["name"]
    assert workflow["user_id"] == str(test_user.id)
    assert len(workflow["nodes"]) == len(template["nodes"])
    assert all(
        node["id"] and node["workflow_id"] == workflow["id"]
        for node in workflow["nodes"]
    )
