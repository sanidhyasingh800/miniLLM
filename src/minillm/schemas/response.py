# a dataclass schema for tracking the responses of a generation request

from dataclasses import dataclass
from .request import Request


# requests contains the resulting request objects with their outputs + metadata
# t_init: the time this batch of requests was requested
# t_end: the time this batch of requests finished all generation and returned to user
@dataclass
class InferenceResponse:
    requests: list[Request]
    t_init: int 
    t_end: int 

    # printing a response object, thanks to gpt
    def __repr__(self):
        total_latency_ms = (self.t_end - self.t_init) / 1e6
        total_time_s = (self.t_end - self.t_init) / 1e9

        total_tokens = sum(req.n for req in self.requests)

        token_throughput = total_tokens / total_time_s if total_time_s > 0 else 0
        request_throughput = len(self.requests) / total_time_s if total_time_s > 0 else 0

        lines = [
            "InferenceResponse(",
            f"  Total Requests: {len(self.requests)}",
            f"  Total Tokens Generated: {total_tokens}",
            f"  Total Latency: {total_latency_ms:.2f} ms",
            f"  Output Throughput: {token_throughput:.2f} tokens/s",
            f"  Request Throughput: {request_throughput:.2f} requests/s",
            "  Requests:"
        ]

        for i, req in enumerate(self.requests):
            ttft = (
                (req.t1 - req.t0) / 1e6
                if req.t0 is not None and req.t1 is not None
                else None
            )

            latency = (
                (req.t2 - req.t0) / 1e6
                if req.t0 is not None and req.t2 is not None
                else None
            )

            tpot = (
                (req.t2 - req.t1) / (req.n - 1) / 1e6
                if req.t1 is not None and req.t2 is not None and req.n > 1
                else None
            )

            lines.extend([
                f"    [{i}] Prompt: {req.prompt!r}",
                f"        Response: {req.response!r}",
                f"        Tokens Generated: {req.n}",
                f"        TTFT: {f'{ttft:.2f} ms' if ttft is not None else 'N/A'}",
                f"        TPOT: {f'{tpot:.2f} ms/token' if tpot is not None else 'N/A'}",
                f"        Latency: {f'{latency:.2f} ms' if latency is not None else 'N/A'}",
            ])

        lines.append(")")
        return "\n".join(lines)