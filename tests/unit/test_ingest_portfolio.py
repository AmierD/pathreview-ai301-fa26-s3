"""Tests for IngestionPipeline.ingest_portfolio and its page fetch."""

import socket

import httpx
import pytest

from ingestion.embeddings.provider import MockEmbeddingProvider
from ingestion.pipeline import MAX_PAGE_BYTES, IngestionPipeline

PORTFOLIO_HTML = """
<html>
<head><title>Jane Doe</title></head>
<body>
  <h1>Jane Doe</h1>
  <h2>About</h2>
  <p>I am a software engineer who builds data tools.</p>
  <h2>Projects</h2>
  <ul><li>Weather App: a React forecast app.</li></ul>
</body>
</html>
"""

# Fake DNS: test hostnames map to fixed IPs, so no real lookups happen.
FAKE_DNS = {
    "portfolio.example.com": "93.184.216.34",
    "internal.example.com": "10.0.0.5",
    "localhost": "127.0.0.1",
}


class FakeVectorDB:
    """Records what the batch processor stores, like a ChromaDB collection."""

    def __init__(self):
        self.ids = []
        self.metadatas = []
        self.documents = []

    def add(self, ids, embeddings, metadatas, documents):
        self.ids.extend(ids)
        self.metadatas.extend(metadatas)
        self.documents.extend(documents)


@pytest.fixture(autouse=True)
def fake_dns(monkeypatch):
    """Resolve hostnames from FAKE_DNS; IP literals resolve to themselves."""

    def getaddrinfo(host, port, *args, **kwargs):
        ip = FAKE_DNS.get(host, host)
        family = socket.AF_INET6 if ":" in ip else socket.AF_INET
        return [(family, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (ip, port))]

    monkeypatch.setattr("ingestion.pipeline.socket.getaddrinfo", getaddrinfo)


def make_pipeline(handler=None, vector_db=None):
    """Build a pipeline whose HTTP requests go to a mock handler instead of the network."""
    transport = httpx.MockTransport(handler) if handler else None
    return IngestionPipeline(
        vector_db=vector_db if vector_db is not None else FakeVectorDB(),
        db_session=None,
        embedding_provider=MockEmbeddingProvider(),
        http_transport=transport,
    )


def html_response(body=PORTFOLIO_HTML):
    return httpx.Response(200, headers={"content-type": "text/html; charset=utf-8"}, text=body)


@pytest.mark.unit
class TestIngestPortfolio:
    """Test suite for IngestionPipeline.ingest_portfolio."""

    def test_ingest_portfolio_stores_chunks(self, monkeypatch):
        """Test that a fetched page is chunked, embedded, and stored with portfolio metadata."""
        vector_db = FakeVectorDB()
        pipeline = make_pipeline(vector_db=vector_db)
        monkeypatch.setattr(pipeline, "_fetch_page", lambda url: PORTFOLIO_HTML)

        result = pipeline.ingest_portfolio("profile-1", "https://portfolio.example.com")

        assert result.skipped is False
        assert result.chunk_count > 0
        assert result.source_id.startswith("portfolio_profile-1_")
        assert len(vector_db.ids) == result.chunk_count
        for metadata in vector_db.metadatas:
            assert metadata["source_type"] == "portfolio"
            assert metadata["url"] == "https://portfolio.example.com"
            assert metadata["profile_id"] == "profile-1"
        assert "software engineer" in " ".join(vector_db.documents)

    def test_ingest_portfolio_without_text_raises(self, monkeypatch):
        """Test that a JavaScript-only page raises and stores nothing."""
        vector_db = FakeVectorDB()
        pipeline = make_pipeline(vector_db=vector_db)
        monkeypatch.setattr(
            pipeline, "_fetch_page", lambda url: '<div id="root"></div><script></script>'
        )

        with pytest.raises(ValueError, match="No extractable text"):
            pipeline.ingest_portfolio("profile-1", "https://portfolio.example.com")

        assert vector_db.ids == []


@pytest.mark.unit
class TestFetchPage:
    """Test suite for IngestionPipeline._fetch_page."""

    def test_fetch_html_page(self):
        """Test that a public HTML page is returned."""
        pipeline = make_pipeline(lambda request: html_response())

        assert "Jane Doe" in pipeline._fetch_page("https://portfolio.example.com")

    @pytest.mark.parametrize(
        "url",
        [
            "ftp://portfolio.example.com/index.html",
            "file:///etc/passwd",
            "http://localhost:8000",
            "http://127.0.0.1",
            "http://169.254.169.254/latest/meta-data/",
            "http://10.0.0.5",
            "http://internal.example.com",
            "http://[::1]/",
        ],
    )
    def test_fetch_refuses_disallowed_urls(self, url):
        """Test that non-http(s) URLs and non-public addresses are refused before any request."""
        requests = []

        def handler(request):
            requests.append(request)
            return html_response()

        pipeline = make_pipeline(handler)

        with pytest.raises(ValueError):
            pipeline._fetch_page(url)

        assert requests == []

    def test_fetch_follows_public_redirect(self):
        """Test that a redirect to another public page is followed."""

        def handler(request):
            if request.url.path == "/":
                return httpx.Response(301, headers={"location": "/home"})
            return html_response()

        pipeline = make_pipeline(handler)

        assert "Jane Doe" in pipeline._fetch_page("https://portfolio.example.com/")

    def test_fetch_refuses_redirect_to_private_ip(self):
        """Test that a redirect to an internal address is refused."""
        requests = []

        def handler(request):
            requests.append(request)
            return httpx.Response(302, headers={"location": "http://169.254.169.254/"})

        pipeline = make_pipeline(handler)

        with pytest.raises(ValueError, match="non-public"):
            pipeline._fetch_page("https://portfolio.example.com")

        assert len(requests) == 1

    def test_fetch_refuses_too_many_redirects(self):
        """Test that a redirect loop stops after the redirect limit."""
        pipeline = make_pipeline(
            lambda request: httpx.Response(302, headers={"location": "/again"})
        )

        with pytest.raises(ValueError, match="Too many redirects"):
            pipeline._fetch_page("https://portfolio.example.com")

    def test_fetch_refuses_non_html(self):
        """Test that a non-HTML content type is refused."""
        pipeline = make_pipeline(
            lambda request: httpx.Response(
                200, headers={"content-type": "application/pdf"}, content=b"%PDF"
            )
        )

        with pytest.raises(ValueError, match="Not an HTML page"):
            pipeline._fetch_page("https://portfolio.example.com")

    def test_fetch_refuses_oversized_page(self):
        """Test that a page over the size limit is refused."""
        big_body = "<p>" + "a" * (MAX_PAGE_BYTES + 1) + "</p>"
        pipeline = make_pipeline(lambda request: html_response(big_body))

        with pytest.raises(ValueError, match="size limit"):
            pipeline._fetch_page("https://portfolio.example.com")

    def test_fetch_raises_on_http_error(self):
        """Test that an error status raises instead of ingesting the error page."""
        pipeline = make_pipeline(lambda request: httpx.Response(404, text="Not found"))

        with pytest.raises(httpx.HTTPStatusError):
            pipeline._fetch_page("https://portfolio.example.com")
