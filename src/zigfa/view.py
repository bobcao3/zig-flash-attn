"""Serves the trace viewer and zone traces on a local port and prints one link per trace.

A link takes the parameters trace, group (sm, cluster, cta, wg or warp), start and end in
microseconds, and expand, the groups to unfold (or all, with fold naming the exceptions), for example
http://127.0.0.1:8732/?trace=traces/0/fwd.jsonl.gz&group=sm&start=100&end=140&expand=sm2.
"""

import argparse
import http.server
import socket
import urllib.parse
import webbrowser
from pathlib import Path

PORT = 8732
CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".gz": "application/gzip",
}


def viewer_files():
    package = Path(__file__).parent
    for directory in (package / "viewer", package.parents[1] / "reports"):
        if (directory / "profile_viewer.html").exists():
            return directory / "profile_viewer.html", directory / "assets" / "report.css"
    raise FileNotFoundError("profile_viewer.html not found; reinstall the package")


def trace_files(paths):
    for path in paths:
        if path.is_dir():
            yield from sorted(path.glob("*.jsonl.gz"))
        elif path.name.endswith(".jsonl.gz") and path.is_file():
            yield path
        else:
            raise SystemExit(f"{path} is neither a .jsonl.gz trace nor a folder of traces")


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        path = self.server.routes.get(urllib.parse.unquote(urllib.parse.urlsplit(self.path).path))
        if path is None:
            self.send_error(404)
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", CONTENT_TYPES[path.suffix])
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


def serve(paths, host="127.0.0.1", port=PORT):
    """Returns a server for the viewer and the traces in `paths`, and the link of each trace. A busy
    port falls back to a free one."""
    viewer, styles = viewer_files()
    traces = list(trace_files(paths))
    if not traces:
        raise SystemExit("no .jsonl.gz traces to serve")
    try:
        server = http.server.ThreadingHTTPServer((host, port), Handler)
    except OSError:
        server = http.server.ThreadingHTTPServer((host, 0), Handler)
    server.routes = {"/": viewer, "/profile_viewer.html": viewer, "/assets/report.css": styles}
    name = socket.gethostname() if host in ("", "0.0.0.0") else host
    links = []
    for index, trace in enumerate(traces):
        route = f"traces/{index}/{trace.name}"
        server.routes[f"/{route}"] = trace.resolve()
        query = urllib.parse.urlencode({"trace": route}, safe="/")
        links.append(f"http://{name}:{server.server_address[1]}/?{query}")
    return server, links


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m zigfa.view",
        description=__doc__.split("\n\n")[0],
        epilog=__doc__.split("\n\n")[1],
    )
    parser.add_argument("traces", nargs="+", type=Path, help=".jsonl.gz traces or folders that hold them")
    parser.add_argument("--host", default="127.0.0.1", help="address to serve on (default: %(default)s)")
    parser.add_argument("--port", type=int, default=PORT, help="port to serve on (default: %(default)s)")
    parser.add_argument("--open", action="store_true", help="open each link in the default browser")
    args = parser.parse_args(argv)
    server, links = serve(args.traces, args.host, args.port)
    for link in links:
        print(link, flush=True)
    if args.open:
        for link in links:
            webbrowser.open_new_tab(link)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
