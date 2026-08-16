"""Local mini server to generate music with MiniMax-Music3 (diffusers in-process).

- Loads the pipeline in a background thread on startup (takes minutes).
- FIFO queue with 1 worker: one generation on the GPU at a time.
- Jobs are kept in memory only; WAVs are saved to ./output/.
"""

import itertools
import os
import queue
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Env-configurable settings.
MODEL_ID = os.environ.get("MUSIC_MODEL_ID", "MiniMaxAI/MiniMax-Music3")  # or a local path from `hf download`
HOST = os.environ.get("MUSIC_HOST", "127.0.0.1")
PORT = int(os.environ.get("MUSIC_PORT", "8000"))
# VRAM strategy ("Low VRAM" recommendations from the model card):
#   auto = auto CPU offload                          -> ~22GB (**default**)
#   low  = auto CPU offload + LLM group offloading -> fits in ~8GB (slower)
#   none = everything on GPU (bf16)                  -> ~24GB
OFFLOAD = os.environ.get("MUSIC_OFFLOAD", "auto")
if OFFLOAD not in {"low", "auto", "none"}:
    raise SystemExit(f"Invalid MUSIC_OFFLOAD: {OFFLOAD!r} (use low, auto or none)")
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
MIN_DURATION, MAX_DURATION = 5.0, 360.0  # seconds (model cap: 9000 frames @ 25 fps = 6 min)

OUTPUT_DIR.mkdir(exist_ok=True)

# --------------------------------------------------------------------------- #
# Estado do modelo
# --------------------------------------------------------------------------- #

model_state = {"pipe": None, "status": "loading", "error": None}  # loading|ready|error
model_ready = threading.Event()


def load_model() -> None:
    try:
        from diffusers import ComponentsManager, ModularPipeline
        from diffusers.hooks import apply_group_offloading

        if OFFLOAD == "none":
            pipe = ModularPipeline.from_pretrained(MODEL_ID)
            pipe.load_components(dtype=torch.bfloat16)
            pipe.to("cuda")
        else:
            manager = ComponentsManager()
            manager.enable_auto_cpu_offload(device="cuda")
            pipe = ModularPipeline.from_pretrained(MODEL_ID, components_manager=manager)
            pipe.load_components(dtype=torch.bfloat16)
            if OFFLOAD == "low":
                # needed below ~22GB — slower, but fits in 8GB
                apply_group_offloading(
                    pipe.language_model,
                    onload_device=torch.device("cuda"),
                    offload_type="leaf_level",
                    use_stream=True,
                )
        model_state["pipe"] = pipe
        model_state["status"] = "ready"
    except Exception as exc:  # noqa: BLE001
        model_state["status"] = "error"
        model_state["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        model_ready.set()


# --------------------------------------------------------------------------- #
# Fila de jobs (FIFO, 1 worker)
# --------------------------------------------------------------------------- #

jobs: dict[str, dict] = {}
job_queue: queue.Queue = queue.Queue()
_seq = itertools.count(1)


class GenerateRequest(BaseModel):
    instructions: str = Field(..., description="Music description (genre, BPM, vocals, arrangement...)")
    lyrics: str = Field("", description="Lyrics with section tags ([Verse], [Chorus], ...) — tags alone on their own line")
    duration: float = Field(60.0, description="Target duration in seconds (5 to 360)")
    seed: int | None = Field(None, description="Seed for reproducibility; empty = random")
    num_inference_steps: int = Field(30, ge=1, le=100, description="Flow-matching steps per window (default 30; more = higher quality but slower)")


def run_generation(job: dict) -> None:
    pipe = model_state["pipe"]
    params = job["params"]

    if params["seed"] is None:
        params["seed"] = int(torch.randint(0, 2**31 - 1, (1,)).item())
    generator = torch.Generator("cuda").manual_seed(params["seed"])

    # output="audios" returns the waveform directly (batch, channels, samples);
    # without it the pipe returns a PipelineState (not subscriptable).
    audios = pipe(
        prompt=params["instructions"],
        lyrics=params["lyrics"],
        audio_duration=params["duration"],
        num_inference_steps=params["num_inference_steps"],
        generator=generator,
        output="audios",
    )
    audio = audios[0]
    if isinstance(audio, torch.Tensor):
        audio = audio.T.float().cpu().numpy()
    else:  # numpy (default output_type is "np")
        audio = audio.T.astype("float32")

    path = OUTPUT_DIR / f"{job['id']}.wav"
    sf.write(path, audio, pipe.sampling_rate)
    job["audio_path"] = str(path)


def worker() -> None:
    model_ready.wait()
    while True:
        job_id = job_queue.get()
        job = jobs.get(job_id)
        if job is None:
            continue
        if model_state["status"] != "ready":
            job["status"] = "error"
            job["error"] = f"Model unavailable: {model_state['error']}"
            continue
        job["status"] = "running"
        job["started_at"] = time.time()
        try:
            run_generation(job)
            job["status"] = "done"
        except Exception as exc:  # noqa: BLE001
            job["status"] = "error"
            job["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            job["finished_at"] = time.time()


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #

@asynccontextmanager
async def lifespan(_: FastAPI):
    threading.Thread(target=load_model, daemon=True).start()
    threading.Thread(target=worker, daemon=True).start()
    yield


app = FastAPI(title="MiniMax Music3 — mini server", lifespan=lifespan)


@app.get("/api/status")
def get_status() -> dict:
    return {
        "model": model_state["status"],
        "model_error": model_state["error"],
        "offload": OFFLOAD,
        "queue_len": job_queue.qsize(),
        "running": any(j["status"] == "running" for j in jobs.values()),
    }


@app.post("/api/generate")
def post_generate(req: GenerateRequest) -> dict:
    if model_state["status"] != "ready":
        raise HTTPException(503, f"Model not ready yet (status: {model_state['status']}). Try again later.")
    instructions = req.instructions.strip()
    if not instructions:
        raise HTTPException(422, "'instructions' cannot be empty.")
    duration = min(max(req.duration, MIN_DURATION), MAX_DURATION)

    job_id = uuid.uuid4().hex[:12]
    jobs[job_id] = {
        "id": job_id,
        "seq": next(_seq),
        "status": "queued",
        "created_at": time.time(),
        "params": {
            "instructions": instructions,
            "lyrics": req.lyrics,
            "duration": duration,
            "seed": req.seed,
            "num_inference_steps": req.num_inference_steps,
        },
        "audio_path": None,
        "error": None,
        "started_at": None,
        "finished_at": None,
    }
    job_queue.put(job_id)
    return {"job_id": job_id}


def queue_position(job: dict) -> int:
    return sum(1 for j in jobs.values() if j["status"] == "queued" and j["seq"] < job["seq"])


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found.")
    out = {
        "id": job["id"],
        "status": job["status"],  # queued|running|done|error
        "params": job["params"],
        "queue_position": queue_position(job),
        "created_at": job["created_at"],
        "started_at": job["started_at"],
        "finished_at": job["finished_at"],
        "error": job["error"],
        "audio_url": f"/api/audio/{job['id']}.wav" if job["status"] == "done" else None,
    }
    return out


@app.get("/api/audio/{filename}")
def get_audio(filename: str) -> FileResponse:
    path = (OUTPUT_DIR / filename).resolve()
    if path.parent != OUTPUT_DIR or not path.exists():
        raise HTTPException(404, "Audio not found.")
    return FileResponse(path, media_type="audio/wav", filename=filename)


# Static UI
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(BASE_DIR / "static" / "index.html")


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
