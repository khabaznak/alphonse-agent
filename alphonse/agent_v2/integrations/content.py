"""Inert bounded MIME evidence. Never fetch images, links or external content."""
from email import policy
from email.parser import BytesParser
from html.parser import HTMLParser
from urllib.parse import urlsplit


class _Text(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.hidden = 0
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in {'script', 'style', 'iframe', 'object', 'template'}:
            self.hidden += 1
        if tag in {'br', 'p', 'div', 'li', 'blockquote'} and not self.hidden:
            self.parts.append('\n')

    def handle_endtag(self, tag):
        if tag in {'script', 'style', 'iframe', 'object', 'template'}:
            self.hidden = max(0, self.hidden - 1)

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def safe_link(url: str) -> str:
    try:
        parsed = urlsplit(url)
        if parsed.scheme == 'https' and parsed.hostname and not parsed.username and not parsed.password:
            return url
    except ValueError:
        pass
    return ''


def normalize_mime(raw: bytes, *, max_bytes: int) -> dict:
    if len(raw) > max_bytes:
        raise ValueError('source_content_budget')
    message = BytesParser(policy=policy.default).parsebytes(raw)
    parts = list(message.walk())
    if len(parts) > 256:
        raise ValueError('source_mime_parts_budget')
    texts, attachments = [], []
    for part in parts:
        if part.is_multipart():
            continue
        kind = part.get_content_type()
        if part.get_content_disposition() == 'attachment' or kind not in {'text/plain', 'text/html'}:
            attachments.append({'content_type': kind, 'requires_authorized_read': True})
            continue
        data = part.get_payload(decode=True) or b''
        try:
            text = data.decode(part.get_content_charset() or 'utf-8', errors='replace')
        except LookupError:
            text = data.decode('utf-8', errors='replace')
        if kind == 'text/html':
            parser = _Text()
            parser.feed(text)
            text = ''.join(parser.parts)
        texts.append({'content_type': kind, 'text': text})
    # Keep quotes and alternative parts as evidence rather than erasing old notices.
    return {'parts': texts, 'attachments': attachments, 'untrusted': True}
