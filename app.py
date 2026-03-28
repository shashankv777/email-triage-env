"""FastAPI application exposing OpenEnv HTTP endpoints for the Email Triage environment."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from env.environment import EmailTriageEnv
from env.models import Email, EmailAction, EmailObservation, EmailReward
from env.tasks import list_tasks


# ---------------------------------------------------------------------------
# Lifespan — initialise the shared environment instance
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager: create the environment on startup."""
    app.state.env = EmailTriageEnv()
    yield


app = FastAPI(
    title="Email Triage OpenEnv",
    version="1.0.0",
    description="An OpenEnv environment for email triage agent training and evaluation.",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / response helpers
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    """Body for POST /reset."""
    task_name: str = "easy"


class StepResponse(BaseModel):
    """Response for POST /step."""
    observation: Dict[str, Any]
    reward: Dict[str, Any]
    done: bool
    info: Dict[str, Any]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/metadata")
async def metadata() -> Dict[str, Any]:
    """Return environment metadata (name, description, version)."""
    return {
        "name": "email-triage-env",
        "description": (
            "An OpenEnv environment for training and evaluating AI agents on "
            "real-world email triage tasks. Agents must classify, prioritise, "
            "and respond to emails across three difficulty levels."
        ),
        "version": "1.0.0",
        "author": "hackathon-participant",
        "tags": ["openenv", "email", "productivity", "nlp", "real-world"],
    }


@app.get("/schema")
async def schema() -> Dict[str, Any]:
    """Return JSON schemas for action, observation, and state."""
    return {
        "action": EmailAction.model_json_schema(),
        "observation": EmailObservation.model_json_schema(),
        "state": {
            "type": "object",
            "description": "Full serialisable state snapshot of the environment",
            "properties": {
                "task": {"type": "string"},
                "step_count": {"type": "integer"},
                "done": {"type": "boolean"},
                "emails": {"type": "array", "items": Email.model_json_schema()},
                "current_email": Email.model_json_schema(),
                "action_history": {"type": "array", "items": {"type": "string"}},
                "replies": {"type": "object"},
                "archived_ids": {"type": "array", "items": {"type": "string"}},
                "opened_ids": {"type": "array", "items": {"type": "string"}},
                "gold_labels": {"type": "object"},
            },
        },
    }


@app.post("/mcp")
async def mcp_endpoint(request: Request) -> JSONResponse:
    """Minimal MCP (JSON-RPC 2.0) endpoint for OpenEnv compatibility."""
    try:
        body = await request.json()
    except Exception:
        body = {}

    rpc_id = body.get("id", 1)
    method = body.get("method", "")

    if method == "initialize":
        result = {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "email-triage-env", "version": "1.0.0"},
        }
    elif method == "tools/list":
        result = {
            "tools": [
                {
                    "name": "reset",
                    "description": "Reset the environment with a task name",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"task_name": {"type": "string", "default": "easy"}},
                    },
                },
                {
                    "name": "step",
                    "description": "Take an action in the environment",
                    "inputSchema": EmailAction.model_json_schema(),
                },
                {
                    "name": "state",
                    "description": "Get the current environment state",
                    "inputSchema": {"type": "object", "properties": {}},
                },
            ]
        }
    elif method == "tools/call":
        params = body.get("params", {})
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})
        env: EmailTriageEnv = app.state.env
        try:
            if tool_name == "reset":
                obs = env.reset(task_name=arguments.get("task_name", "easy"))
                result = {"content": [{"type": "text", "text": obs.model_dump_json()}]}
            elif tool_name == "step":
                action = EmailAction(**arguments)
                obs, reward, done, info = env.step(action)
                result = {"content": [{"type": "text", "text": EmailReward.model_validate(reward).model_dump_json()}]}
            elif tool_name == "state":
                result = {"content": [{"type": "text", "text": str(env.state())}]}
            else:
                result = {"content": [{"type": "text", "text": f"Unknown tool: {tool_name}"}], "isError": True}
        except Exception as exc:
            result = {"content": [{"type": "text", "text": str(exc)}], "isError": True}
    else:
        result = {}

    return JSONResponse({"jsonrpc": "2.0", "id": rpc_id, "result": result})


@app.get("/tasks")
async def tasks() -> list[dict]:
    """Return the list of available tasks and descriptions."""
    return list_tasks()


@app.post("/reset")
async def reset(body: ResetRequest) -> Dict[str, Any]:
    """Reset the environment and start a new episode.

    Args:
        body: JSON with task_name ('easy', 'medium', or 'hard').

    Returns:
        The initial EmailObservation as a JSON dict.
    """
    env: EmailTriageEnv = app.state.env
    try:
        obs: EmailObservation = env.reset(task_name=body.task_name)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return obs.model_dump()


@app.post("/step")
async def step(action: EmailAction) -> Dict[str, Any]:
    """Execute one action in the environment.

    Args:
        action: The agent's EmailAction.

    Returns:
        A dict with observation, reward, done, and info.
    """
    env: EmailTriageEnv = app.state.env
    try:
        obs, reward, done, info = env.step(action)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
    }


@app.get("/state")
async def state() -> Dict[str, Any]:
    """Return the full serialisable state snapshot."""
    env: EmailTriageEnv = app.state.env
    return env.state()
