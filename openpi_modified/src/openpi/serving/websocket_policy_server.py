import asyncio
import http
import logging
import time
import traceback
import numpy as np

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)


def unpackb_with_writable(data):
    try:
        obj = msgpack_numpy.unpackb(data)
    except Exception as e:
        logger.error(f"Failed to unpack msgpack data: {e}")
        raise ValueError(f"Invalid msgpack data: {e}") from e

    def _make_writable(obj):
        if isinstance(obj, np.ndarray):
            obj = obj.copy()
            obj.setflags(write=1)
        elif isinstance(obj, dict):
            for k, v in obj.items():
                obj[k] = _make_writable(v)
        elif isinstance(obj, list) or isinstance(obj, tuple):
            obj = type(obj)(_make_writable(item) for item in obj)
        return obj

    try:
        return _make_writable(obj)
    except Exception as e:
        logger.error(f"Failed to make unpacked data writable: {e}")
        raise RuntimeError(f"Failed to process unpacked data: {e}") from e


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Implements the `load`, `infer`, and `score` methods.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
        *,
        value_temperature: float = 1.0,
        enable_score_endpoint: bool = True,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._value_temperature = value_temperature

        # Check if policy supports score_observation AND endpoint is enabled
        self._supports_score = (
            enable_score_endpoint
            and hasattr(self._policy, "score_observation")
        )
        self._metadata["supports_score_endpoint"] = self._supports_score

        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                obs = unpackb_with_writable(await websocket.recv())

                # Extract request type (default "infer" for backward compatibility)
                request_type = obs.get("_request_type", "infer")
                obs = {k: v for k, v in obs.items() if k != "_request_type"}

                if request_type not in ("infer", "score"):
                    raise ValueError(
                        f"Invalid _request_type: {request_type}. "
                        f"Must be 'infer' or 'score'."
                    )

                process_time = time.monotonic()

                if request_type == "score":
                    if not self._supports_score:
                        raise NotImplementedError(
                            "score_observation is not supported by the current policy. "
                            "Please use a model with RL value head enabled."
                        )
                    result = self._policy.score_observation(
                        obs, value_temperature=self._value_temperature
                    )
                else:
                    result = self._policy.infer(obs)

                process_time = time.monotonic() - process_time

                result["server_timing"] = {
                    "process_ms": process_time * 1000,
                }
                if prev_total_time is not None:
                    result["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(result))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception as e:
                logger.error(f"Error handling request from {websocket.remote_address}: {e}")
                logger.debug(traceback.format_exc())
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(
    connection: _server.ServerConnection, request: _server.Request
) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None
