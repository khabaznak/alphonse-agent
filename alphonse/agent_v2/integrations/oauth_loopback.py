"""Local OAuth browser handoff. Callback bytes never enter IPC logs or history."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
from urllib.parse import parse_qs, urlsplit

from alphonse.agent_v2.accounts import AccountSession


class OAuthLoopback:
    def __init__(self, oauth):
        self.oauth = oauth
        self.servers = {}
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
                            except Exception:
                                pass
                    self.send_response(status)
                    self.send_header('Content-Type', 'text/plain; charset=utf-8')
                    self.send_header('Cache-Control', 'no-store')
                    self.send_header('Referrer-Policy', 'no-referrer')
                    self.end_headers()
                    self.wfile.write(message.encode())

            server = ThreadingHTTPServer(('127.0.0.1', uri.port), Handler)
            self.servers[uri.port] = server
            threading.Thread(target=server.serve_forever, daemon=True, name='oauth-loopback').start()

    def stop(self):
        with self.lock:
            for server in self.servers.values():
                server.shutdown()
                server.server_close()
            self.servers.clear()
