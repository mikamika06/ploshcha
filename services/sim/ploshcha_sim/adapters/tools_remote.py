import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

from ..domain.evidence import found_in
from ..ports.tool import ToolCall, ToolPort, ToolResult, ToolSpec

DEFAULT_TIMEOUT_S = 20.0

Transport = Callable[[str, dict], object]


class RemoteToolbox(ToolPort):
    def __init__(self, manifest: list[dict], transport: Transport, *,
                 timeout_s: float = DEFAULT_TIMEOUT_S):
        self._specs = [
            ToolSpec(name=str(entry["name"]),
                     description=str(entry.get("description") or entry["name"]),
                     params=dict(entry.get("params") or entry.get("parameters") or
                                 {"type": "object", "properties": {}}))
            for entry in manifest
        ]
        self._by_name = {s.name: s for s in self._specs}
        self._transport = transport
        self.timeout_s = timeout_s
        self.calls: list[tuple[str, dict]] = []
        self._pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="remote-tool")

    def specs(self) -> list[ToolSpec]:
        return list(self._specs)

    def call(self, request: ToolCall) -> ToolResult:
        if request.tool not in self._by_name:
            return ToolResult(tool=request.tool, ok=False, error="unknown_remote_tool")
        self.calls.append((request.tool, dict(request.args)))
        t0 = time.perf_counter()
        try:
            future = self._pool.submit(self._transport, request.tool, dict(request.args))
            value = future.result(timeout=self.timeout_s)
        except FutureTimeout:
            return ToolResult(tool=request.tool, ok=False, error="remote_timeout",
                              latency_ms=int((time.perf_counter() - t0) * 1000))
        except Exception as exc:
            return ToolResult(tool=request.tool, ok=False,
                              error=f"remote_error: {type(exc).__name__}: {exc}"[:300],
                              latency_ms=int((time.perf_counter() - t0) * 1000))
        latency = int((time.perf_counter() - t0) * 1000)
        if isinstance(value, dict) and value.get("error"):
            return ToolResult(tool=request.tool, ok=False, error=str(value["error"])[:300],
                              latency_ms=latency)
        return ToolResult(tool=request.tool, ok=True, value=value,
                          found=found_in(value), latency_ms=latency)
