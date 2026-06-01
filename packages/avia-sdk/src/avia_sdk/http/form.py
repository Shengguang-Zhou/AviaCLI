from __future__ import annotations

import json
from typing import Any
from urllib import error as urlerror, parse, request

from avia_sdk.auth.tokens import refresh_after_auth_error
from avia_sdk.errors import _AviaHTTPError


def _request_form_json(
    *,
    method: str,
    url: str,
    token: str,
    fields: dict[str, object],
    timeout: int | float = 60,
) -> dict[str, Any]:
    data = parse.urlencode({key: str(value) for key, value in fields.items()}).encode()
    for auth_attempt in range(2):
        req = request.Request(
            url,
            data=data,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            method=method.upper(),
        )
        try:
            with request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
            break
        except urlerror.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:4000]
            error = _AviaHTTPError(
                method=method,
                url=url,
                status=int(exc.code),
                reason=str(exc.reason),
                detail=detail,
            )
            if auth_attempt == 0 and refresh_after_auth_error(token, error, label=method):
                continue
            raise error from exc
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))
