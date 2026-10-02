import threading
import webbrowser

from .server import App, make_server


class Bridge:
    def __init__(self):
        self._window = None

    def pick_folder(self):
        import webview

        dialog = getattr(getattr(webview, "FileDialog", None), "FOLDER", None)
        if dialog is None:
            dialog = webview.FOLDER_DIALOG
        result = self._window.create_file_dialog(dialog)
        if not result:
            return None
        return result[0] if isinstance(result, (list, tuple)) else result


def launch(app: App, host: str = "127.0.0.1", port: int = 8765) -> None:
    server = make_server(app, host, port)
    host, port = server.server_address[:2]
    url = f"http://{host}:{port}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"CodeRetriever backend at {url} (device: {app.device})")
    try:
        import webview
    except ImportError:
        print("pywebview is not installed, opening the app in your browser instead")
        webbrowser.open(url)
        try:
            thread.join()
        except KeyboardInterrupt:
            pass
        finally:
            server.shutdown()
        return
    bridge = Bridge()
    bridge._window = webview.create_window(
        "CodeRetriever",
        url,
        width=1440,
        height=920,
        min_size=(900, 640),
        js_api=bridge,
        background_color="#0b0b0b",
    )
    webview.start()
    server.shutdown()
