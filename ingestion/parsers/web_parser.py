import re
from html.parser import HTMLParser

from .base import BaseParser, ParseResult

# Elements whose text is never page content (code, styling, chrome).
SKIP_TAGS = {"script", "style", "noscript", "template", "svg", "nav", "footer"}

# Headings kept as markdown so the text keeps its section structure.
HEADING_TAGS = {"h1": 1, "h2": 2, "h3": 3}

# Elements that start a new line of text.
BLOCK_TAGS = {
    "address",
    "article",
    "aside",
    "blockquote",
    "br",
    "dd",
    "div",
    "dl",
    "dt",
    "figcaption",
    "figure",
    "form",
    "h4",
    "h5",
    "h6",
    "header",
    "hr",
    "li",
    "main",
    "ol",
    "p",
    "pre",
    "section",
    "table",
    "td",
    "th",
    "tr",
    "ul",
}


class _TextExtractor(HTMLParser):
    """Collect visible text from HTML, keeping headings and line breaks."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title_parts: list[str] = []
        self.skip_depth = 0
        self.in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in SKIP_TAGS:
            self.skip_depth += 1
        elif tag == "title":
            self.in_title = True
        elif tag in HEADING_TAGS:
            self.parts.append("\n" + "#" * HEADING_TAGS[tag] + " ")
        elif tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_startendtag(self, tag, attrs):
        # Self-closing tags (<br/>, <hr/>) never contain text, so don't open a skip region.
        if tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in SKIP_TAGS:
            if self.skip_depth:
                self.skip_depth -= 1
        elif tag == "title":
            self.in_title = False
        elif tag in HEADING_TAGS or tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        if self.skip_depth:
            return
        if self.in_title:
            self.title_parts.append(data)
        else:
            # Newlines in HTML source are just whitespace; only tags start new lines.
            self.parts.append(re.sub(r"\s+", " ", data))


class WebParser(BaseParser):
    """Parser for portfolio web pages (HTML)."""

    def parse(self, content: str | bytes) -> ParseResult:
        """
        Parse a web page from HTML content.

        Args:
            content: HTML string or bytes

        Returns:
            ParseResult with extracted text and metadata

        Raises:
            ValueError: If content is not a valid string/bytes, or has no extractable text
        """
        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="replace")
        elif not isinstance(content, str):
            raise ValueError("Content must be a string or bytes")

        extractor = _TextExtractor()
        extractor.feed(content)
        extractor.close()

        text = self._clean_text("".join(extractor.parts))
        if not text:
            raise ValueError("No extractable text")

        title = " ".join("".join(extractor.title_parts).split())
        heading_count = len(re.findall(r"^#{1,3} ", text, flags=re.MULTILINE))

        metadata = {
            "source_type": "portfolio",
            "title": title,
            "heading_count": heading_count,
            "word_count": len(text.split()),
        }

        return ParseResult(
            text=text,
            metadata=metadata,
            source_type="portfolio",
        )

    def _clean_text(self, text: str) -> str:
        """Collapse whitespace within lines and drop empty lines and empty headings."""
        lines = []
        for line in text.split("\n"):
            line = " ".join(line.split())
            if line and not re.fullmatch(r"#{1,3}", line):
                lines.append(line)
        return "\n".join(lines)
