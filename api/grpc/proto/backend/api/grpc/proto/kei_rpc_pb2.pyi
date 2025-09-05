# Type stubs for generated protobuf classes

from google.protobuf import message as _message

class Pagination(_message.Message):
    page: int
    per_page: int
    total: int
    def __init__(self, *, page: int = ..., per_page: int = ..., total: int = ...) -> None: ...

class Resource(_message.Message):
    id: str
    name: str
    created_at: str
    updated_at: str
    def __init__(self, *, id: str = ..., name: str = ..., created_at: str = ..., updated_at: str = ...) -> None: ...

class ListResourcesRequest(_message.Message):
    page: int
    per_page: int
    q: str
    sort: str
    def __init__(self, *, page: int = ..., per_page: int = ..., q: str = ..., sort: str = ...) -> None: ...

class ListResourcesResponse(_message.Message):
    items: list[Resource]
    pagination: Pagination | None
    def __init__(self, *, items: list[Resource] = ..., pagination: Pagination | None = ...) -> None: ...

class CreateResourceRequest(_message.Message):
    name: str
    idempotency_key: str
    def __init__(self, *, name: str = ..., idempotency_key: str = ...) -> None: ...

class GetResourceRequest(_message.Message):
    id: str
    def __init__(self, *, id: str = ...) -> None: ...

class PatchResourceRequest(_message.Message):
    id: str
    name: str
    if_match: str
    def __init__(self, *, id: str = ..., name: str = ..., if_match: str = ...) -> None: ...

class BatchCreateRequest(_message.Message):
    items: list[CreateResourceRequest]
    def __init__(self, *, items: list[CreateResourceRequest] = ...) -> None: ...

class StreamMessage(_message.Message):
    id: str
    type: str
    operation: str
    payload_json: str
    def __init__(self, *, id: str = ..., type: str = ..., operation: str = ..., payload_json: str = ...) -> None: ...

class PlanRequest(_message.Message):
    objective: str
    context_json: str
    def __init__(self, *, objective: str = ..., context_json: str = ...) -> None: ...

class ActRequest(_message.Message):
    action: str
    context_json: str
    def __init__(self, *, action: str = ..., context_json: str = ...) -> None: ...

class ObserveRequest(_message.Message):
    observation: str
    context_json: str
    def __init__(self, *, observation: str = ..., context_json: str = ...) -> None: ...

class ExplainRequest(_message.Message):
    topic: str
    context_json: str
    def __init__(self, *, topic: str = ..., context_json: str = ...) -> None: ...

class OperationResponse(_message.Message):
    operation: str
    status: str
    started_at: str
    completed_at: str
    result_json: str
    error: str
    def __init__(self, *, operation: str = ..., status: str = ..., started_at: str = ..., completed_at: str = ..., result_json: str = ..., error: str = ...) -> None: ...
