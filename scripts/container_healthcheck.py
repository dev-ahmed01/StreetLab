"""Authenticated loopback readiness probe for the container image."""
from __future__ import annotations

import base64
import json
import os
import urllib.request

password = os.environ["STREETLAB_ACCESS_PASSWORD"]
encoded = base64.b64encode(("streetlab:" + password).encode()).decode()
request = urllib.request.Request(
    "http://127.0.0.1:8000/api/health",
    headers={"Authorization": "Basic " + encoded},
)
with urllib.request.urlopen(request, timeout=5) as response:
    assert response.status == 200
    result = json.load(response)
    assert result["status"] == "ok" and result["product_role"] == "decision_assurance"
