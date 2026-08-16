"""MCP server (stdio) exposing the MiniMax Music3 mini server as MCP tools.

A thin client over the HTTP API — server.py must be running (default
http://127.0.0.1:8000). Tools follow the job pattern: `generate_music`
returns a job_id immediately (generation takes minutes); poll `get_job`
until status is done|error.
"""

import os

import httpx
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

HOST = os.environ.get("MUSIC_HOST", "127.0.0.1")
PORT = os.environ.get("MUSIC_PORT", "8000")
BASE_URL = os.environ.get("MUSIC_BASE_URL", f"http://{HOST}:{PORT}")

mcp = FastMCP("minimax-music")


def _call(method: str, path: str, **kwargs) -> httpx.Response:
    try:
        return httpx.request(method, f"{BASE_URL}{path}", timeout=30, **kwargs)
    except httpx.HTTPError as exc:
        raise ToolError(
            f"Music server unreachable at {BASE_URL} — is server.py running? ({exc})"
        ) from exc


@mcp.tool
def get_status() -> dict:
    """Server/model status: model (loading|ready|error), offload mode, queue length, running?

    Call this first: on startup the model takes minutes to load — wait until
    model == "ready" before generating.
    """
    return _call("GET", "/api/status").json()


@mcp.tool
def generate_music(
    instructions: str,
    lyrics: str = "",
    duration: float = 60.0,
    seed: int | None = None,
    num_inference_steps: int = 30,
) -> str:
    """Enqueue a music generation job and return its job_id immediately
    (does NOT wait — a song takes minutes). Then poll get_job(job_id) until
    status is "done" or "error"; the finished WAV is at the returned audio_url.

    Args:
        instructions: Music description — genre, BPM, key, vocals, arrangement, mood.
        lyrics: Optional lyrics with section tags each on their own line
            ([Intro], [Verse], [Chorus], ...).
        duration: Target duration in seconds (5-360).
        seed: Seed for reproducibility; omit for random.
        num_inference_steps: Flow-matching steps per window (1-100, default 30;
            more = higher quality but slower).
    """
    r = _call(
        "POST",
        "/api/generate",
        json={
            "instructions": instructions,
            "lyrics": lyrics,
            "duration": duration,
            "seed": seed,
            "num_inference_steps": num_inference_steps,
        },
    )
    if r.status_code != 200:
        raise ToolError(str(r.json().get("detail", r.text)))
    return r.json()["job_id"]


@mcp.tool
def get_job(job_id: str) -> dict:
    """Poll a generation job: status (queued|running|done|error), queue_position,
    params, error, and audio_url (absolute, set when status is "done")."""
    r = _call("GET", f"/api/jobs/{job_id}")
    if r.status_code == 404:
        raise ToolError(f"Unknown job_id: {job_id}")
    out = r.json()
    if out.get("audio_url"):
        out["audio_url"] = BASE_URL + out["audio_url"]  # make it absolute
    return out


if __name__ == "__main__":
    mcp.run()  # stdio transport
