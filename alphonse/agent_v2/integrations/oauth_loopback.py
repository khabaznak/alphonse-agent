"""Local OAuth browser handoff. Callback bytes never enter IPC logs or history."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import time
from urllib.parse import parse_qs, urlsplit

from alphonse.agent_v2.accounts import AccountSession
from alphonse.agent_v2.integrations.oauth import safe_callback_message


class _TrackedHTTPServer(ThreadingHTTPServer):
    """Keep callbacks visible until they have finished writing credentials."""
    def __init__(self, *args, **kwargs):
        self._callback_condition = threading.Condition()
        self._callbacks = 0
        super().__init__(*args, **kwargs)

    def process_request(self, request, client_address):
        with self._callback_condition:
            self._callbacks += 1
        try:
            super().process_request(request, client_address)
        except Exception:
            with self._callback_condition:
                self._callbacks -= 1
                self._callback_condition.notify_all()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            with self._callback_condition:
                self._callbacks -= 1
                self._callback_condition.notify_all()

    def wait_quiescent(self, timeout=None):
        with self._callback_condition:
            return self._callback_condition.wait_for(lambda: self._callbacks == 0, timeout)


class OAuthLoopback:
    def __init__(self, oauth):
        self.oauth = oauth
        self.servers = {}
        self._retired_servers = []
        self.lock = threading.RLock()

    def start(self, provider):
        client = self.oauth.clients.get(provider)
        if not client:
            raise ValueError('oauth_provider_not_configured')
        redirect = client.config.redirect_uri
        uri = urlsplit(redirect)
        if uri.scheme != 'http' or uri.hostname != '127.0.0.1':
            raise ValueError('oauth_desktop_loopback_required')
        with self.lock:
            if uri.port in self.servers:
                return
            owner = self

            class Handler(BaseHTTPRequestHandler):
                def log_message(self, *args):
                    pass

                def do_GET(self):
                    path = urlsplit(self.path)
                    status = 400
                    message = 'Authorization could not be completed. Return to Alphonse and reconnect.'
                    redirects = [c.config.redirect_uri for c in owner.oauth.clients.values()
                        if urlsplit(c.config.redirect_uri).scheme == 'http'
                        and urlsplit(c.config.redirect_uri).hostname == '127.0.0.1'
                        and urlsplit(c.config.redirect_uri).port == uri.port
                        and urlsplit(c.config.redirect_uri).path == path.path]
                    if redirects:
                        state = parse_qs(path.query).get('state', [''])[0]
                        with owner.oauth.lock:
                            pending = owner.oauth.pending.get(state)
                        if pending:
                            try:
                                owner.oauth.complete(AccountSession(pending.actor), redirects[0] + '?' + path.query)
                                status = 200
                                message = 'Account connected. You can close this window and return to Alphonse.'
                            except Exception as exc:
                                message = safe_callback_message(exc)
                    self.send_response(status)
                    self.send_header('Content-Type', 'text/plain; charset=utf-8')
                    self.send_header('Cache-Control', 'no-store')
                    self.send_header('Referrer-Policy', 'no-referrer')
                    self.end_headers()
                    self.wfile.write(message.encode())

            server = _TrackedHTTPServer(('127.0.0.1', uri.port), Handler)
            try:
                threading.Thread(target=server.serve_forever, daemon=True, name='oauth-loopback').start()
            except Exception:
                server.server_close()
                raise
            self.servers[uri.port] = server

    def stop(self):
        with self.lock:
            for server in self.servers.values():
                server.shutdown()
                server.server_close()
                self._retired_servers.append(server)
            self.servers.clear()

    def wait_quiescent(self, timeout=None):
        # The IPC lifecycle prevents new starts after stop, so this list is
        # stable for shutdown.
        with self.lock:
            servers = tuple(self._retired_servers)
            if self.servers:
                return False
        deadline = None if timeout is None else time.monotonic() + max(0, timeout)
        return all(server.wait_quiescent(None if deadline is None else max(0, deadline - time.monotonic()))
                   for server in servers)
