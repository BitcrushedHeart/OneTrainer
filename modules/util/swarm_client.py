"""Minimal SwarmUI HTTP API client.

Only what the off-policy generator needs: a session, the model list, and
text2image. SwarmUI writes each generated image into its own output folder
(``.../Text2Img/local/raw/<date>/``) with its usual metadata, so nothing is
downloaded here — the returned ``View/...`` paths are enough to report what
was made, and the off-policy builder picks the files up on its next scan.

The API is documented at
https://github.com/mcmonkeyprojects/SwarmUI/blob/master/docs/API.md
"""

import logging
import threading

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:7801"
#: A generation call blocks until the images exist; a queued batch on a busy
#: GPU can sit for minutes, so this is deliberately generous.
DEFAULT_TIMEOUT = 900


class SwarmError(RuntimeError):
    """A SwarmUI call failed. ``error_id`` is set when the API supplied one."""

    def __init__(self, message: str, error_id: str | None = None):
        super().__init__(message)
        self.error_id = error_id


def normalize_base_url(base_url: str) -> str:
    url = (base_url or DEFAULT_BASE_URL).strip().rstrip("/")
    if not url:
        return DEFAULT_BASE_URL
    if "://" not in url:
        url = "http://" + url
    return url


class SwarmClient:
    """Thread-safe-enough SwarmUI client: one session, re-acquired on demand.

    SwarmUI invalidates sessions on restart and reports that as
    ``error_id == "invalid_session_id"``; every call retries once through a
    fresh session rather than surfacing that to the user.
    """

    def __init__(self, base_url: str = DEFAULT_BASE_URL, token: str | None = None, timeout: int = DEFAULT_TIMEOUT):
        self.base_url = normalize_base_url(base_url)
        self.token = (token or "").strip() or None
        self.timeout = timeout
        self._session_id: str | None = None
        self._lock = threading.Lock()
        self._http = None

    # ------------------------------------------------------------------ #

    def _client(self):
        if self._http is None:
            import requests

            self._http = requests.Session()
            if self.token:
                self._http.cookies.set("swarm_token", self.token)
        return self._http

    def _post(self, route: str, payload: dict, timeout: int | None = None) -> dict:
        import requests

        url = f"{self.base_url}/API/{route}"
        try:
            response = self._client().post(url, json=payload, timeout=timeout or self.timeout)
        except requests.RequestException as ex:
            raise SwarmError(f"Could not reach SwarmUI at {self.base_url}: {ex}") from ex
        try:
            data = response.json()
        except ValueError as ex:
            raise SwarmError(f"SwarmUI returned a non-JSON response ({response.status_code}) for {route}") from ex
        if not isinstance(data, dict):
            raise SwarmError(f"SwarmUI returned an unexpected response for {route}")
        error_id = data.get("error_id")
        error = data.get("error")
        if error_id or error:
            raise SwarmError(str(error or error_id), error_id=error_id if isinstance(error_id, str) else None)
        return data

    def session_id(self, refresh: bool = False) -> str:
        with self._lock:
            if refresh or not self._session_id:
                data = self._post("GetNewSession", {})
                session = data.get("session_id")
                if not session:
                    raise SwarmError("SwarmUI did not return a session id")
                self._session_id = str(session)
            return self._session_id

    def _call(self, route: str, payload: dict, timeout: int | None = None) -> dict:
        """POST with the session injected, retrying once on a stale session."""
        body = dict(payload)
        body["session_id"] = self.session_id()
        try:
            return self._post(route, body, timeout)
        except SwarmError as ex:
            if ex.error_id != "invalid_session_id":
                raise
            body["session_id"] = self.session_id(refresh=True)
            return self._post(route, body, timeout)

    # ------------------------------------------------------------------ #

    def ping(self) -> dict:
        """Confirm SwarmUI is reachable; returns its version and user."""
        data = self._post("GetNewSession", {})
        with self._lock:
            self._session_id = str(data.get("session_id") or "") or None
        return {"version": data.get("version"), "user_id": data.get("user_id")}

    def list_loaded_models(self) -> list[str]:
        """Checkpoints SwarmUI currently has loaded — the natural default, since
        it is what the user was just generating with in the web UI."""
        data = self._call("ListLoadedModels", {})
        models = data.get("models")
        if not isinstance(models, list):
            return []
        return [str(entry.get("name")) for entry in models if isinstance(entry, dict) and entry.get("name")]

    def presets(self) -> list[dict]:
        """The user's saved parameter presets, newest API shape: each has a
        ``title`` and a ``param_map`` of generation parameters. SwarmUI keeps a
        ``default`` preset, which is the closest server-side record of the
        settings the user generates with."""
        data = self._call("GetMyUserData", {})
        presets = data.get("presets")
        if not isinstance(presets, list):
            return []
        out = []
        for entry in presets:
            if not isinstance(entry, dict):
                continue
            params = entry.get("param_map")
            if isinstance(params, dict):
                out.append({"title": str(entry.get("title") or "(unnamed)"), "params": params})
        return out

    def list_models(self, subtype: str = "Stable-Diffusion", depth: int = 3) -> list[str]:
        """Model names usable as the ``model`` generation parameter."""
        data = self._call("ListModels", {"path": "", "depth": depth, "subtype": subtype})
        files = data.get("files")
        if not isinstance(files, list):
            return []
        names = []
        for entry in files:
            name = entry.get("name") if isinstance(entry, dict) else None
            if isinstance(name, str) and name:
                names.append(name)
        return sorted(names)

    def generate(
        self,
        prompt: str,
        width: int,
        height: int,
        model: str,
        images: int = 1,
        steps: int | None = None,
        cfg_scale: float | None = None,
        seed: int = -1,
        negative_prompt: str = "",
        loras: list[str] | None = None,
        lora_weights: list[float] | None = None,
        extra_params: dict | None = None,
        timeout: int | None = None,
    ) -> list[str]:
        """Generate ``images`` images and return SwarmUI's ``View/...`` paths.

        ``seed = -1`` asks SwarmUI for a random seed, which is what a batch of
        negatives wants — several samples of the same prompt should differ.
        Width and height are passed through exactly: the API takes any size,
        unlike the web UI's fixed aspect list, so callers can request a
        trainer bucket directly.
        """
        payload: dict = {
            "prompt": prompt,
            "model": model,
            "images": max(1, int(images)),
            "width": int(width),
            "height": int(height),
            "seed": int(seed),
        }
        if steps is not None:
            payload["steps"] = int(steps)
        if cfg_scale is not None:
            payload["cfgscale"] = float(cfg_scale)
        if negative_prompt:
            payload["negativeprompt"] = negative_prompt
        if loras:
            # SwarmUI pairs these positionally; a missing weight defaults to 1.
            payload["loras"] = list(loras)
            weights = list(lora_weights or [])
            weights += [1.0] * (len(loras) - len(weights))
            payload["loraweights"] = [float(weight) for weight in weights[: len(loras)]]
        if extra_params:
            payload.update(extra_params)
        data = self._call("GenerateText2Image", payload, timeout)
        result = data.get("images")
        if not isinstance(result, list):
            raise SwarmError("SwarmUI returned no images")
        return [str(path) for path in result]
