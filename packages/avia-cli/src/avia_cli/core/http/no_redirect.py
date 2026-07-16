from __future__ import annotations

import json
from urllib import error as urlerror, request


class UnexpectedHTTPRedirect(RuntimeError):
    def __init__(self, *, method: str, url: str, status: int, location: str) -> None:
        self.method = str(method).upper()
        self.url = str(url)
        self.status = int(status)
        self.location = str(location)
        super().__init__(
            json.dumps(
                {
                    "code": "unexpected_http_redirect",
                    "location": self.location,
                    "method": self.method,
                    "status": self.status,
                    "url": self.url,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )


class _RejectRedirectHandler(request.HTTPRedirectHandler):
    def redirect_request(self, *_args, **_kwargs):
        return None


_OPENER = request.build_opener(_RejectRedirectHandler())


def open_no_redirect(req: request.Request, *, timeout: float):
    try:
        return _OPENER.open(req, timeout=timeout)
    except urlerror.HTTPError as exc:
        if 300 <= int(exc.code) < 400:
            location = str(exc.headers.get("Location") or "") if exc.headers else ""
            raise UnexpectedHTTPRedirect(
                method=req.get_method(),
                url=req.full_url,
                status=int(exc.code),
                location=location,
            ) from exc
        raise
