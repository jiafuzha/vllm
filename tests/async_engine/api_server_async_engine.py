"""vllm.entrypoints.api_server with some extra logging for testing."""
import argparse
from typing import Any, Dict

import uvicorn
from fastapi.responses import JSONResponse, Response

from vllm.extension import ns as ns

import vllm.entrypoints.api_server
from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine

app = vllm.entrypoints.api_server.app


class AsyncLLMEngineWithStats(AsyncLLMEngine):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._num_aborts = 0

    async def abort(self, request_id: str) -> None:
        await super().abort(request_id)
        self._num_aborts += 1

    def testing_stats(self) -> Dict[str, Any]:
        return {"num_aborted_requests": self._num_aborts}


@app.get("/stats")
def stats() -> Response:
    """Get the statistics of the engine."""
    return JSONResponse(engine.testing_stats())

def _modify_qunatization_choices(parser, dest, choices):
    for action in parser._actions:
        if action.dest == dest:
            action.choices = choices
            return
    else:
        raise ValueError("argument {} not found".format(dest))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8070)
    parser = AsyncEngineArgs.add_cli_args(parser)
    _modify_qunatization_choices(parser, "quantization", ('awq', 'gptq', 'squeezellm', 'ns', None))
    _modify_qunatization_choices(parser, "block_size", None)

    args = parser.parse_args()

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngineWithStats.from_engine_args(engine_args)
    vllm.entrypoints.api_server.engine = engine
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="debug",
        timeout_keep_alive=vllm.entrypoints.api_server.TIMEOUT_KEEP_ALIVE)
