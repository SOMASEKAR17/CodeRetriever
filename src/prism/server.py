import json
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from urllib.parse import parse_qs, urlparse


class App:
    def __init__(self, engine_factory, device: str):
        self.engine_factory = engine_factory
        self.device = device
        self.engine = engine_factory(device)
        self.lock = threading.RLock()
        self.jobs: dict[str, dict] = {}

    def settings(self) -> dict:
        embedder = self.engine.embedder
        try:
            import torch

            gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        except Exception:
            gpu = None
        return {
            "device": self.device,
            "gpu": gpu,
            "space": embedder.space,
            "dim": embedder.dim,
            "model": getattr(embedder, "model_name", embedder.space),
            "reranker": bool(self.engine.reranker),
            "index_dir": str(self.engine.index_dir),
        }

    def benchmarks(self) -> list[dict]:
        from .config import REPO_ROOT

        path = REPO_ROOT / "benchmarks.json"
        if not path.exists():
            return []
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def set_device(self, device: str) -> dict:
        with self.lock:
            if device != self.device:
                self.engine.store.close()
                self.engine = self.engine_factory(device)
                self.device = device
        return self.settings()

    def sources(self) -> list[dict]:
        with self.lock:
            return [
                {"id": sid, "kind": kind, "location": location, "versions": self.engine.versions(sid)}
                for sid, kind, location in self.engine.store.sources()
            ]

    def available_versions(self, source_id: str) -> list[dict]:
        with self.lock:
            return [{"label": v.label, "ref": v.ref, "ordinal": v.ordinal} for v in self.engine.available_versions(source_id)]

    def start_job(self, kind: str, target) -> str:
        job_id = uuid.uuid4().hex[:8]
        job = {"id": job_id, "kind": kind, "status": "running", "log": [], "result": None, "started": time.time()}
        self.jobs[job_id] = job

        def progress(message: str) -> None:
            job["log"].append(message)

        def run() -> None:
            try:
                with self.lock:
                    job["result"] = target(progress)
                job["status"] = "done"
            except Exception as exc:
                job["status"] = "failed"
                job["log"].append(f"{type(exc).__name__}: {exc}")
                job["log"].append(traceback.format_exc())
            job["seconds"] = round(time.time() - job["started"], 2)

        threading.Thread(target=run, daemon=True).start()
        return job_id

    def add_source(self, body: dict) -> str:
        location = body["location"].strip()
        source_id = body.get("id") or None

        def target(progress):
            from pathlib import Path

            if location.startswith(("http://", "https://", "git@")) or (Path(location) / ".git").exists():
                progress(f"registering git source {location}")
                sid = self.engine.add_git(location, source_id, partial=bool(body.get("partial")))
            else:
                progress(f"registering folder {location}")
                sid = self.engine.add_folder(location, source_id)
            progress(f"added {sid}")
            return {"source": sid}

        return self.start_job("add", target)

    def index(self, source_id: str, refs: list[str] | None) -> str:
        def target(progress):
            stats = self.engine.index(source_id, refs or None, progress=progress)
            return [s.as_dict() for s in stats]

        return self.start_job("index", target)

    def search(self, body: dict) -> dict:
        query, source_id = body["query"], body["source"]
        version, k = body.get("version") or "all", int(body.get("k") or 10)
        started = time.time()
        with self.lock:
            if version == "all":
                intent, hits = self.engine.search_evolution(query, source_id, k)
                payload = {"mode": intent.kind, "matched": intent.matched, "hits": [h.__dict__ for h in hits]}
            else:
                hits = self.engine.search(query, source_id, version, k)
                payload = {"mode": "version", "matched": version, "hits": [h.__dict__ for h in hits]}
        payload["ms"] = round((time.time() - started) * 1000, 1)
        return payload


def make_handler(app: App):
    page = resources.files("prism").joinpath("static/index.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            return

        def _send(self, code: int, payload, content_type: str = "application/json") -> None:
            data = payload if isinstance(payload, bytes) else json.dumps(payload, default=str).encode()
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _body(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(length) or b"{}")

        def _route(self, method: str) -> None:
            url = urlparse(self.path)
            parts = [p for p in url.path.split("/") if p]
            query = parse_qs(url.query)
            try:
                if method == "GET" and not parts:
                    return self._send(200, page, "text/html; charset=utf-8")
                if parts[:1] != ["api"]:
                    return self._send(404, {"error": "not found"})
                route = parts[1:]
                if method == "GET" and route == ["benchmarks"]:
                    return self._send(200, app.benchmarks())
                if method == "GET" and route == ["settings"]:
                    return self._send(200, app.settings())
                if method == "POST" and route == ["settings"]:
                    return self._send(200, app.set_device(self._body()["device"]))
                if method == "GET" and route == ["sources"]:
                    return self._send(200, app.sources())
                if method == "POST" and route == ["sources"]:
                    return self._send(202, {"job": app.add_source(self._body())})
                if method == "GET" and len(route) == 3 and route[0] == "sources" and route[2] == "available":
                    return self._send(200, app.available_versions(route[1]))
                if method == "POST" and len(route) == 3 and route[0] == "sources" and route[2] == "index":
                    return self._send(202, {"job": app.index(route[1], self._body().get("refs"))})
                if method == "GET" and len(route) == 2 and route[0] == "jobs":
                    job = app.jobs.get(route[1])
                    return self._send(200 if job else 404, job or {"error": "unknown job"})
                if method == "POST" and route == ["search"]:
                    return self._send(200, app.search(self._body()))
                return self._send(404, {"error": "not found"})
            except KeyError as exc:
                return self._send(400, {"error": f"missing or unknown: {exc}"})
            except Exception as exc:
                return self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

        def do_GET(self):
            self._route("GET")

        def do_POST(self):
            self._route("POST")

    return Handler


def make_server(app: App, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    last_error = None
    for candidate in range(port, port + 20):
        try:
            return ThreadingHTTPServer((host, candidate), make_handler(app))
        except OSError as exc:
            last_error = exc
    raise last_error


def serve(app: App, host: str = "127.0.0.1", port: int = 8765) -> None:
    server = make_server(app, host, port)
    host, port = server.server_address[:2]
    print(f"CodeRetriever running at http://{host}:{port}  (device: {app.device}, Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
