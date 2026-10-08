"""Tests for web_parser.py"""

import pytest

from ingestion.parsers.base import ParseResult
from ingestion.parsers.web_parser import WebParser

PORTFOLIO_HTML = """
<!doctype html>
<html>
<head>
  <title>  Jane Doe | Portfolio  </title>
  <style>body { color: red; }</style>
  <script>window.analytics = "tracking-code";</script>
</head>
<body>
  <nav><a href="/">Home</a><a href="/blog">Blog</a></nav>
  <h1>Jane Doe</h1>
  <section>
    <h2>About</h2>
    <p>I am a   software engineer
       who builds data tools.</p>
  </section>
  <section>
    <h2>Projects</h2>
    <ul>
      <li>Weather App: a React forecast app.</li>
      <li>PathFinder: a route planner in Go.</li>
    </ul>
    <svg><text>chart label</text></svg>
  </section>
  <footer>Copyright 2026 Jane Doe</footer>
</body>
</html>
"""


@pytest.mark.unit
class TestWebParser:
    """Test suite for WebParser."""

    @pytest.fixture
    def parser(self):
        """Create a WebParser instance."""
        return WebParser()

    def test_parse_portfolio_keeps_content(self, parser):
        """Test that bio and project text are kept."""
        result = parser.parse(PORTFOLIO_HTML)

        assert isinstance(result, ParseResult)
        assert result.source_type == "portfolio"
        assert "I am a software engineer who builds data tools." in result.text
        assert "Weather App: a React forecast app." in result.text
        assert "PathFinder: a route planner in Go." in result.text

    def test_parse_drops_non_content_elements(self, parser):
        """Test that script, style, nav, svg, and footer text is removed."""
        result = parser.parse(PORTFOLIO_HTML)

        assert "tracking-code" not in result.text
        assert "color: red" not in result.text
        assert "Blog" not in result.text
        assert "chart label" not in result.text
        assert "Copyright" not in result.text

    def test_parse_headings_become_markdown(self, parser):
        """Test that h1/h2 become markdown heading lines and are counted."""
        result = parser.parse(PORTFOLIO_HTML)
        lines = result.text.split("\n")

        assert "# Jane Doe" in lines
        assert "## About" in lines
        assert "## Projects" in lines
        assert result.metadata["heading_count"] == 3

    def test_parse_title_in_metadata(self, parser):
        """Test that <title> goes into metadata, not the body text."""
        result = parser.parse(PORTFOLIO_HTML)

        assert result.metadata["title"] == "Jane Doe | Portfolio"
        assert "Portfolio" not in result.text

    def test_parse_collapses_whitespace(self, parser):
        """Test that runs of whitespace and blank lines are collapsed."""
        result = parser.parse(PORTFOLIO_HTML)

        assert "  " not in result.text
        assert "\n\n" not in result.text
        assert result.text == result.text.strip()

    def test_metadata_structure(self, parser):
        """Test that metadata has required structure."""
        result = parser.parse("<p>Hello world</p>")

        assert result.metadata["source_type"] == "portfolio"
        assert result.metadata["title"] == ""
        assert result.metadata["heading_count"] == 0
        assert result.metadata["word_count"] == 2

    def test_parse_bytes_input(self, parser):
        """Test parsing HTML from UTF-8 bytes input."""
        result = parser.parse("<p>Café projects 🎉</p>".encode())

        assert "Café projects" in result.text

    def test_parse_invalid_content_type(self, parser):
        """Test that invalid content type raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            parser.parse(12345)

        assert "Content must be a string or bytes" in str(exc_info.value)

    def test_parse_javascript_only_page_raises(self, parser):
        """Test that a client-rendered page with no text raises ValueError."""
        spa_html = """
        <html>
        <head><title>My App</title><script src="/bundle.js"></script></head>
        <body>
          <noscript>You need to enable JavaScript to run this app.</noscript>
          <div id="root"></div>
        </body>
        </html>
        """
        with pytest.raises(ValueError) as exc_info:
            parser.parse(spa_html)

        assert "No extractable text" in str(exc_info.value)
