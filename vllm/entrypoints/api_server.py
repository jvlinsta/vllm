"""
NOTE: This API server is used only for demonstrating usage of AsyncEngine
and simple performance benchmarks. It is not intended for production use.
For production use, we recommend using our OpenAI compatible server.
We are also not going to accept PRs modifying this file, please
change `vllm/entrypoints/openai/api_server.py` instead.
"""

import os
from pathlib import Path
import torch
import json
import ssl
from typing import AsyncGenerator
import gc
import ray
from ray import tune

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.usage.usage_lib import UsageContext
from vllm.utils import FlexibleArgumentParser, random_uuid
from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment

#DEV: to switch into HF hub models
from huggingface_hub import repo_exists

os.environ[
    'VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'  #https://github.com/vllm-project/vllm/issues/6152

logger = init_logger("vllm.entrypoints.api_server")

TIMEOUT_KEEP_ALIVE = 5  # seconds.
app = FastAPI()
engine = None


def get_model_files(model_path):
    bin_files = list(model_path.glob("pytorch_model*.bin"))
    sft_files = list(model_path.glob("model*.safetensors"))
    return bin_files + sft_files


def get_model_path(model_path):
    if repo_exists(model_path):  #valid huggingface repo
        return model_path
    model_path = Path(model_path)
    if len(get_model_files(model_path)) > 0:
        return str(model_path)
    merged_path = model_path / "merged"
    if len(get_model_files(merged_path)) > 0:
        return str(merged_path)
    raise ValueError("Incorrect model path")


@app.post("/change_model")
async def change_model(request: Request) -> Response:
    request_dict = await request.json()
    new_path = request_dict.pop("model_path")
    print(f"Request with new path: {new_path}")
    try:
        new_path = get_model_path(
            new_path)  #just checking if we should get merged or not
    except Exception as e:
        return JSONResponse(status_code=405,
                            content={"message": str(e) + ' cannot get path'})

    print(f"Setting  new path: {new_path}")
    global engine_args
    current_path = engine_args.model
    if current_path == new_path:
        return Response(status_code=200)

    # DEV: https://github.com/vllm-project/vllm/issues/1908
    destroy_model_parallel()  ##Invalid process group specified
    destroy_distributed_environment()
    global engine
    del engine
    gc.collect()
    # del llm.llm_engine.model_executor
    # del llm
    gc.collect()
    torch.cuda.empty_cache()
    #ray.shutdown()

    print(f"cuda memory: {torch.cuda.memory_allocated() // 1024 // 1024}MB")

    engine_args.model = new_path
    engine_args.tokenizer = new_path

    try:
        return Response(status_code=200)
    except Exception as e:
        return JSONResponse(
            status_code=404,
            content={"message": str(e) + ' cannot change model'})


@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


@app.post("/generate")
async def generate(request: Request) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    request_dict = await request.json()
    prompt = request_dict.pop("prompt")
    stream = request_dict.pop("stream", False)
    sampling_params = SamplingParams(**request_dict)
    request_id = random_uuid()

    assert engine is not None
    results_generator = engine.generate(prompt, sampling_params, request_id)

    # Streaming case
    async def stream_results() -> AsyncGenerator[bytes, None]:
        async for request_output in results_generator:
            prompt = request_output.prompt
            text_outputs = [
                prompt + output.text for output in request_output.outputs
            ]
            ret = {"text": text_outputs}
            yield (json.dumps(ret) + "\0").encode("utf-8")

    if stream:
        return StreamingResponse(stream_results())

    # Non-streaming case
    final_output = None
    async for request_output in results_generator:
        if await request.is_disconnected():
            # Abort the request if the client disconnects.
            await engine.abort(request_id)
            return Response(status_code=499)
        final_output = request_output

    assert final_output is not None
    prompt = final_output.prompt
    text_outputs = [prompt + output.text for output in final_output.outputs]
    ret = {"text": text_outputs}
    return JSONResponse(ret)


if __name__ == "__main__":
    parser = FlexibleArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--ssl-keyfile", type=str, default=None)
    parser.add_argument("--ssl-certfile", type=str, default=None)
    parser.add_argument("--ssl-ca-certs",
                        type=str,
                        default=None,
                        help="The CA certificates file")
    parser.add_argument(
        "--ssl-cert-reqs",
        type=int,
        default=int(ssl.CERT_NONE),
        help="Whether client certificate is required (see stdlib ssl module's)"
    )
    parser.add_argument(
        "--root-path",
        type=str,
        default=None,
        help="FastAPI root_path when app is behind a path based routing proxy")
    parser.add_argument("--log-level", type=str, default="debug")
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(
        engine_args, usage_context=UsageContext.API_SERVER)

    app.root_path = args.root_path

    logger.info("Available routes are:")
    for route in app.routes:
        if not hasattr(route, 'methods'):
            continue
        methods = ', '.join(route.methods)
        logger.info("Route: %s, Methods: %s", route.path, methods)

    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level=args.log_level,
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
                ssl_keyfile=args.ssl_keyfile,
                ssl_certfile=args.ssl_certfile,
                ssl_ca_certs=args.ssl_ca_certs,
                ssl_cert_reqs=args.ssl_cert_reqs)
