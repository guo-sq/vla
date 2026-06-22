import logging
import time
from typing import Dict, Optional, Tuple

from typing_extensions import override
import websockets.sync.client

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy


class WebsocketClientPolicy(_base_policy.BasePolicy):
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: Optional[int] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = msgpack_numpy.Packer()
        self._api_key = api_key
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info(f"Waiting for server at {self._uri}...")
        while True:
            try:
                headers = (
                    {"Authorization": f"Api-Key {self._api_key}"}
                    if self._api_key
                    else None
                )
                conn = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    additional_headers=headers,
                )
                metadata = msgpack_numpy.unpackb(conn.recv())
                return conn, metadata
            except ConnectionRefusedError:
                logging.info("Still waiting for server...")
                time.sleep(5)

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        try:
            data = self._packer.pack(obs)
            self._ws.send(data)
            response = self._ws.recv()
            if isinstance(response, str):
                # we're expecting bytes; if the server sends a string, it's an error.
                raise RuntimeError(f"Error in inference server:\n{response}")
            return msgpack_numpy.unpackb(response)
        except ConnectionError as e:
            raise RuntimeError(f"Connection to server {self._uri} failed during infer: {e}") from e
        except TimeoutError as e:
            raise RuntimeError(f"Timeout waiting for server response from {self._uri}: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error during infer: {e}") from e

    @override
    def score_observation(self, obs: Dict) -> Dict:  # noqa: UP006
        """Request value score from server.

        Adds _request_type="score" to the observation before sending.
        The server routes to score_observation instead of infer.

        Returns:
            dict with:
                - "value": scalar value (float)
                - "value_logits": raw logits from model (np.ndarray)
                - "value_metadata": dict with value_bins, value_range, is_distributional
                - "policy_timing": timing info from policy
                - "server_timing": timing info from server

        Raises:
            RuntimeError: If connection fails, times out, or server returns error.
        """
        try:
            obs_with_type = {**obs, "_request_type": "score"}
            data = self._packer.pack(obs_with_type)
            self._ws.send(data)
            response = self._ws.recv()
            if isinstance(response, str):
                raise RuntimeError(f"Error in inference server:\n{response}")
            return msgpack_numpy.unpackb(response)
        except ConnectionError as e:
            raise RuntimeError(f"Connection to server {self._uri} failed during score: {e}") from e
        except TimeoutError as e:
            raise RuntimeError(f"Timeout waiting for server response from {self._uri}: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error during score_observation: {e}") from e

    @override
    def reset(self) -> None:
        pass
