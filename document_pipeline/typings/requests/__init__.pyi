from typing import Any

from requests.auth import HTTPBasicAuth

class Response:
    status_code: int

    def json(self) -> Any: ...
    def raise_for_status(self) -> None: ...

def post(
    url: str,
    data: dict[str, str],
    auth: HTTPBasicAuth | None = ...,
    headers: dict[str, str] | None = ...,
    timeout: int | float | None = ...,
    verify: bool = ...,
) -> Response: ...
