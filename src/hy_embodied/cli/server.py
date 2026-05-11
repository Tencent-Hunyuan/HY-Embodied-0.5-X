"""OpenAI-compatible HTTP server for HY-Embodied-0.5-X.

Exposes ``/v1/chat/completions`` (and ``/v1/models``) so that any OpenAI SDK
client can seamlessly talk to HY-Embodied.

Usage::

    python -m hy_embodied.cli.server --model ckpts/HY-Embodied-0.5-X

    # or after `pip install -e .`
    hy-embodied-server --model ckpts/HY-Embodied-0.5-X --port 8080

Then call it with the standard OpenAI Python client::

    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:8080/v1", api_key="any")
    resp = client.chat.completions.create(
        model="HY-Embodied-0.5-X",
        messages=[
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}},
                {"type": "text", "text": "Describe this image."},
            ]},
        ],
    )
    print(resp.choices[0].message.content)
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import logging
import os
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from hy_embodied.inference import GenerationConfig, HyEmbodiedPipeline

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global pipeline (populated on startup)
# ---------------------------------------------------------------------------
_pipeline: HyEmbodiedPipeline | None = None
_model_name: str = "HY-Embodied-0.5-X"

# ---------------------------------------------------------------------------
# Pydantic request / response schemas (OpenAI-compatible subset)
# ---------------------------------------------------------------------------


class ImageURL(BaseModel):
    url: str
    detail: str | None = None


class ContentPart(BaseModel):
    type: str  # "text" | "image_url"
    text: str | None = None
    image_url: ImageURL | None = None


class ChatMessage(BaseModel):
    role: str
    content: str | list[ContentPart]


class ChatCompletionRequest(BaseModel):
    model: str = "HY-Embodied-0.5-X"
    messages: list[ChatMessage]
    temperature: float = 0.05
    max_tokens: int | None = Field(default=None, alias="max_tokens")
    max_completion_tokens: int | None = None
    stream: bool = False
    enable_thinking: bool | None = None  # custom extension


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: dict
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: UsageInfo


class DeltaMessage(BaseModel):
    role: str | None = None
    content: str | None = None


class ChatCompletionStreamChoice(BaseModel):
    index: int = 0
    delta: DeltaMessage
    finish_reason: str | None = None


class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionStreamChoice]


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "tencent"


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelCard]


# ---------------------------------------------------------------------------
# Helper: convert OpenAI messages to pipeline-native format
# ---------------------------------------------------------------------------

def _resolve_image(url: str) -> str:
    """Convert an OpenAI ``image_url`` value to a path or URL the pipeline accepts.

    Supports:
      - data:image/...;base64,... → temp file path
      - http(s)://... → passed through
      - local file path → passed through
    """
    if url.startswith("data:"):
        # data URI → decode → temp file
        header, b64data = url.split(",", 1)
        ext = ".png"
        if "jpeg" in header or "jpg" in header:
            ext = ".jpg"
        elif "webp" in header:
            ext = ".webp"
        raw = base64.b64decode(b64data)
        tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
        tmp.write(raw)
        tmp.flush()
        tmp.close()
        return tmp.name
    return url  # http(s) or local path


def _openai_messages_to_pipeline(messages: list[ChatMessage]) -> tuple[list[dict], list[str]]:
    """Transform OpenAI-style messages into the format expected by
    ``HyEmbodiedPipeline.build_messages``.

    Returns ``(pipeline_messages, temp_files)`` where ``temp_files`` lists
    paths of base64-decoded temp images that should be cleaned up after
    inference.
    """
    pipeline_msgs: list[dict] = []
    temp_files: list[str] = []

    for msg in messages:
        if isinstance(msg.content, str):
            pipeline_msgs.append({"role": msg.role, "content": [{"type": "text", "text": msg.content}]})
        else:
            content_parts: list[dict] = []
            for part in msg.content:
                if part.type == "text" and part.text:
                    content_parts.append({"type": "text", "text": part.text})
                elif part.type == "image_url" and part.image_url:
                    resolved = _resolve_image(part.image_url.url)
                    if resolved != part.image_url.url:
                        temp_files.append(resolved)
                    content_parts.append({"type": "image", "image": resolved})
            pipeline_msgs.append({"role": msg.role, "content": content_parts})

    return pipeline_msgs, temp_files


def _cleanup_temp_files(paths: list[str]) -> None:
    for p in paths:
        try:
            os.unlink(p)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Core inference (runs in thread to avoid blocking the event loop)
# ---------------------------------------------------------------------------

def _run_inference(
    messages: list[dict],
    gen_cfg: GenerationConfig,
) -> str:
    """Synchronous inference using the global pipeline."""
    assert _pipeline is not None
    inputs = _pipeline.processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=gen_cfg.enable_thinking,
    ).to(_pipeline.model.device)

    with torch.no_grad():
        generated_ids = _pipeline.model.generate(
            **inputs,
            max_new_tokens=gen_cfg.max_new_tokens,
            use_cache=gen_cfg.use_cache,
            temperature=gen_cfg.temperature,
            do_sample=gen_cfg.temperature > 0,
        )

    output_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    return _pipeline.processor.batch_decode(
        output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]


def _run_inference_stream(
    messages: list[dict],
    gen_cfg: GenerationConfig,
) -> str:
    """Same as _run_inference but returns the full text (streaming is
    simulated at the HTTP layer by chunking the output)."""
    return _run_inference(messages, gen_cfg)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    app = FastAPI(title="HY-Embodied-0.5-X OpenAI API", version="0.5.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ---- /v1/models ----
    @app.get("/v1/models")
    async def list_models():
        return ModelList(data=[ModelCard(id=_model_name, created=int(time.time()))])

    # ---- /v1/chat/completions ----
    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        if _pipeline is None:
            raise HTTPException(status_code=503, detail="Model not loaded yet")

        pipeline_msgs, temp_files = _openai_messages_to_pipeline(request.messages)

        max_tokens = request.max_completion_tokens or request.max_tokens or 32768
        enable_thinking = request.enable_thinking if request.enable_thinking is not None else False

        gen_cfg = GenerationConfig(
            max_new_tokens=max_tokens,
            temperature=request.temperature,
            enable_thinking=enable_thinking,
        )

        request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        try:
            if request.stream:
                return await _handle_stream(request_id, created, pipeline_msgs, gen_cfg, temp_files)
            else:
                return await _handle_non_stream(request_id, created, pipeline_msgs, gen_cfg, temp_files)
        except Exception as e:
            _cleanup_temp_files(temp_files)
            logger.exception("Inference failed")
            raise HTTPException(status_code=500, detail=str(e))

    async def _handle_non_stream(
        request_id: str,
        created: int,
        messages: list[dict],
        gen_cfg: GenerationConfig,
        temp_files: list[str],
    ) -> JSONResponse:
        loop = asyncio.get_event_loop()
        output = await loop.run_in_executor(None, _run_inference, messages, gen_cfg)
        _cleanup_temp_files(temp_files)

        response = ChatCompletionResponse(
            id=request_id,
            created=created,
            model=_model_name,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message={"role": "assistant", "content": output},
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
            ),
        )
        return JSONResponse(content=response.model_dump())

    async def _handle_stream(
        request_id: str,
        created: int,
        messages: list[dict],
        gen_cfg: GenerationConfig,
        temp_files: list[str],
    ) -> StreamingResponse:
        loop = asyncio.get_event_loop()
        output = await loop.run_in_executor(None, _run_inference_stream, messages, gen_cfg)
        _cleanup_temp_files(temp_files)

        async def event_generator() -> AsyncIterator[str]:
            # First chunk: role
            chunk = ChatCompletionStreamResponse(
                id=request_id,
                created=created,
                model=_model_name,
                choices=[
                    ChatCompletionStreamChoice(
                        index=0,
                        delta=DeltaMessage(role="assistant", content=""),
                        finish_reason=None,
                    )
                ],
            )
            yield f"data: {chunk.model_dump_json()}\n\n"

            # Content chunks (split by ~20 chars for a streaming feel)
            chunk_size = 20
            for i in range(0, len(output), chunk_size):
                text_piece = output[i : i + chunk_size]
                chunk = ChatCompletionStreamResponse(
                    id=request_id,
                    created=created,
                    model=_model_name,
                    choices=[
                        ChatCompletionStreamChoice(
                            index=0,
                            delta=DeltaMessage(content=text_piece),
                            finish_reason=None,
                        )
                    ],
                )
                yield f"data: {chunk.model_dump_json()}\n\n"
                await asyncio.sleep(0)  # yield control

            # Final chunk: finish_reason
            chunk = ChatCompletionStreamResponse(
                id=request_id,
                created=created,
                model=_model_name,
                choices=[
                    ChatCompletionStreamChoice(
                        index=0,
                        delta=DeltaMessage(),
                        finish_reason="stop",
                    )
                ],
            )
            yield f"data: {chunk.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    # ---- Health check ----
    @app.get("/health")
    async def health():
        return {"status": "ok", "model_loaded": _pipeline is not None}

    @app.get("/")
    async def root():
        return {
            "message": "HY-Embodied-0.5-X OpenAI-compatible API server",
            "docs": "/docs",
            "endpoints": ["/v1/models", "/v1/chat/completions", "/health"],
        }

    return app


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch an OpenAI-compatible API server for HY-Embodied-0.5-X"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="tencent/HY-Embodied-0.5-X",
        help="Model directory or Hugging Face Hub repo id.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--workers", type=int, default=1, help="Number of uvicorn workers (1 recommended for GPU)")
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Name reported in /v1/models. Defaults to the basename of --model.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model precision (default: bfloat16).",
    )
    return parser.parse_args()


def main() -> None:
    global _pipeline, _model_name

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args()

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }

    _model_name = args.model_name or Path(args.model).name or "HY-Embodied-0.5-X"

    logger.info("Loading model %s on %s (%s) ...", args.model, args.device, args.dtype)
    _pipeline = HyEmbodiedPipeline.from_pretrained(
        args.model,
        device=args.device,
        torch_dtype=dtype_map[args.dtype],
    )
    logger.info("Model loaded. Starting server on %s:%d", args.host, args.port)

    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port, workers=args.workers)


if __name__ == "__main__":
    main()
