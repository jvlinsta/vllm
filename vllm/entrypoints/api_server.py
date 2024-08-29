"""
NOTE: This API server is used only for demonstrating usage of AsyncEngine
and simple performance benchmarks. It is not intended for production use.
For production use, we recommend using our OpenAI compatible server.
We are also not going to accept PRs modifying this file, please
change `vllm/entrypoints/openai/api_server.py` instead.
"""
import os
import gc
from pathlib import Path
import torch
import asyncio
import json
import ssl
from argparse import Namespace
from typing import Any, AsyncGenerator, Optional
import contextlib

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

import vllm
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.launcher import serve_http
from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.usage.usage_lib import UsageContext
from vllm.utils import FlexibleArgumentParser, random_uuid, is_cpu, iterate_with_cancellation
from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment

#DEV: to switch into HF hub models
from huggingface_hub import repo_exists
from vllm.version import __version__ as VLLM_VERSION

# os.environ[
#     'VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'  #https://github.com/vllm-project/vllm/issues/6152

logger = init_logger("vllm.entrypoints.api_server")

TIMEOUT_KEEP_ALIVE = 5  # seconds.
app = FastAPI()
engine = None
vllm.engine.async_llm_engine._raise_exception_on_finish = lambda task, error_callback: None


def get_free_gpus():
    free_gpus = []
    num_gpus = torch.cuda.device_count()

    for i in range(num_gpus):
        # Check the current device
        device = torch.device(f'cuda:{i}')
        memory_allocated = torch.cuda.memory_allocated(device)

        if memory_allocated == 0:
            free_gpus.append(str(i))
    print(f"Free GPUs: {free_gpus}")
    return free_gpus


def set_cuda_visible_devices(free_gpus):
    if free_gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(free_gpus)
        print(
            f"CUDA_VISIBLE_DEVICES set to {os.environ['CUDA_VISIBLE_DEVICES']}"
        )
    else:
        print("No free GPUs with 0 memory allocated found.")


def cleanup():
    destroy_model_parallel()
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()
    gc.collect()
    if not is_cpu():
        torch.cuda.empty_cache()


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

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # DEV: https://github.com/vllm-project/vllm/issues/1908
    global engine  # global to ensure we are deleting the engine?
    #need to delete every attribute of engine
    logger.warning(engine.__dict__)
    # kill the (otherwise `while True`) background loop of your AsyncLLMEngine
    # engine._background_loop_unshielded.cancel()
    del engine.engine.model_executor
    del engine
    #https://discuss.pytorch.org/t/cuda-memory-not-released-by-torch-cuda-empty-cache/129913/6
    cleanup()
    # for obj in gc.get_objects():
    #     if torch.is_tensor(obj):
    #         logger.warning(type(obj), obj.size())
    #     try:
    #         if (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
    #             logger.info(type(obj.data), obj.data.size())
    #     except Exception as e:
    #         logger.warning(e)
    gc.collect()

    #for each cuda device print memory
    # for i in range(torch.cuda.device_count()):
    #     print(torch.cuda.memory_summary(i))
    free_gpus = get_free_gpus()
    #set_cuda_visible_devices(free_gpus)
    '''
      File "<string>", line 15, in __init__
    File "/home/ubuntu/vllm/vllm/config.py", line 1620, in __post_init__
        self.model_config.verify_with_parallel_config(self.parallel_config)
    File "/home/ubuntu/vllm/vllm/config.py", line 272, in verify_with_parallel_config
        raise ValueError(
    ValueError: Total number of attention heads (16) must be divisible by tensor parallel size (3).
    '''

    #engine_args.tensor_parallel_size = len(free_gpus) - 1 #HARDCODED fix ... dropping to 6

    # parallel config

    engine_args.model = new_path
    engine_args.tokenizer = new_path
    engine_args.distributed_executor_backend = 'mp'
    engine = AsyncLLMEngine.from_engine_args(
        engine_args, usage_context=UsageContext.API_SERVER)
    logger.info('New engine setup')
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
    results_generator = iterate_with_cancellation(
        results_generator, is_cancelled=request.is_disconnected)

    # Streaming case
    async def stream_results() -> AsyncGenerator[bytes, None]:
        async for request_output in results_generator:
            prompt = request_output.prompt
            assert prompt is not None
            text_outputs = [
                prompt + output.text for output in request_output.outputs
            ]
            ret = {"text": text_outputs}
            yield (json.dumps(ret) + "\0").encode("utf-8")

    if stream:
        return StreamingResponse(stream_results())

    # Non-streaming case
    final_output = None
    try:
        async for request_output in results_generator:
            final_output = request_output
    except asyncio.CancelledError:
        return Response(status_code=499)

    assert final_output is not None
    prompt = final_output.prompt
    assert prompt is not None
    text_outputs = [prompt + output.text for output in final_output.outputs]
    ret = {"text": text_outputs}
    return JSONResponse(ret)


def build_app(args: Namespace) -> FastAPI:
    global app

    app.root_path = args.root_path
    return app


async def init_app(
    args: Namespace,
    llm_engine: Optional[AsyncLLMEngine] = None,
) -> FastAPI:
    app = build_app(args)

    global engine

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = (llm_engine
              if llm_engine is not None else AsyncLLMEngine.from_engine_args(
                  engine_args, usage_context=UsageContext.API_SERVER))

    return app


async def run_server(args: Namespace,
                     llm_engine: Optional[AsyncLLMEngine] = None,
                     **uvicorn_kwargs: Any) -> None:
    logger.info("vLLM API server version %s", VLLM_VERSION)
    logger.info("args: %s", args)

    app = await init_app(args, llm_engine)
    assert engine is not None

    shutdown_task = await serve_http(
        app,
        engine=engine,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile,
        ssl_ca_certs=args.ssl_ca_certs,
        ssl_cert_reqs=args.ssl_cert_reqs,
        **uvicorn_kwargs,
    )

    await shutdown_task


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

    asyncio.run(run_server(args))
