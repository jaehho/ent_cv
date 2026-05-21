"""Custom middleware for the api app."""

from django.middleware.gzip import GZipMiddleware


class JsonGZipMiddleware(GZipMiddleware):
    """Compress only JSON responses.

    The vanilla GZipMiddleware also compresses StreamingHttpResponse, which
    breaks video range requests: it strips Content-Length and recompresses
    already-compressed mp4 bytes, so the browser stalls into a buffering loop
    instead of progressive playback. Detection payloads (the actual win — 50–
    100 MB JSON per case) are JSON, so gating compression on Content-Type
    gives us the wire savings without touching the video stream.
    """

    def process_response(self, request, response):
        content_type = response.get("Content-Type", "")
        if "application/json" not in content_type.lower():
            return response
        return super().process_response(request, response)
