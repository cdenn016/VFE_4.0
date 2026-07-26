"""Disabled-by-default click launcher for the full H6 candidate producer."""

from __future__ import annotations

import hashlib
import hmac
import json


CONFIG = {
    "enabled": False,
    "authorization": None,
    "config": {},
}


def main() -> dict[str, object]:
    enabled = CONFIG.get("enabled")
    if type(enabled) is not bool:
        raise ValueError("enabled must be an exact boolean")
    if not enabled:
        result: dict[str, object] = {
            "operation": "H6-Validation-Perturbations",
            "status": "IDLE",
        }
        print(
            json.dumps(
                result,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return result

    authorization = CONFIG.get("authorization")
    phrase = "AUTHORIZE_VFE4_H6_VALIDATION_PERTURBATIONS_4096_V1"
    expected_sha256 = (
        "6a2c61ad2f1ad7fdeb798dee8be231b6ff1393290ad77ac0bd262f2d49da88ae"
    )
    if (
        type(authorization) is not str
        or not hmac.compare_digest(authorization, phrase)
        or not hmac.compare_digest(
            hashlib.sha256(authorization.encode("utf-8")).hexdigest(),
            expected_sha256,
        )
    ):
        raise PermissionError(
            "exact H6 validation perturbation authorization is required"
        )
    raw_config = CONFIG.get("config")
    if type(raw_config) is not dict:
        raise ValueError("authorized config must be an exact dictionary")

    from verification.h6_validation_candidate import (
        run_h6_validation_perturbation_build,
    )

    result = run_h6_validation_perturbation_build(raw_config)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return result


if __name__ == "__main__":
    main()
