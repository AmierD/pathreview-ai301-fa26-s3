import hashlib
import ipaddress
import socket
from dataclasses import dataclass
from urllib.parse import urlsplit

import httpx
import structlog

from .chunking.strategy_selector import StrategySelector
from .embeddings.batch_processor import BatchEmbeddingProcessor
from .embeddings.provider import EmbeddingProvider
from .parsers.readme_parser import ReadmeParser
from .parsers.repo_analyzer import RepoAnalyzer
from .parsers.resume_parser import ResumeParser
from .parsers.web_parser import WebParser

logger = structlog.get_logger()

# Limits for fetching a portfolio page.
FETCH_TIMEOUT_SECONDS = 10.0
MAX_PAGE_BYTES = 2 * 1024 * 1024
MAX_REDIRECTS = 3


@dataclass
class IngestResult:
    """Result of ingesting a source."""

    source_id: str
    chunk_count: int
    skipped: bool
    skip_reason: str | None = None


class IngestionPipeline:
    """Main orchestration for document ingestion and embedding."""

    def __init__(
        self,
        vector_db,
        db_session,
        embedding_provider: EmbeddingProvider,
        http_transport: httpx.BaseTransport | None = None,
    ):
        """
        Initialize the ingestion pipeline.

        Args:
            vector_db: ChromaDB collection for storing embeddings
            db_session: Database session for storing metadata
            embedding_provider: EmbeddingProvider for generating embeddings
            http_transport: Optional httpx transport for fetching web pages (tests use a mock)
        """
        self.vector_db = vector_db
        self.db_session = db_session
        self.embedding_provider = embedding_provider
        self.http_transport = http_transport
        self.strategy_selector = StrategySelector()
        self.batch_processor = BatchEmbeddingProcessor(embedding_provider, vector_db)

        # Initialize parsers
        self.resume_parser = ResumeParser()
        self.readme_parser = ReadmeParser()
        self.repo_analyzer = RepoAnalyzer()
        self.web_parser = WebParser()

    def ingest_resume(
        self,
        profile_id: str,
        content: str | bytes,
        filename: str,
    ) -> IngestResult:
        """
        Ingest a resume document.

        Args:
            profile_id: ID of the profile owner
            content: Resume content (PDF bytes or markdown string)
            filename: Original filename

        Returns:
            IngestResult with ingestion status
        """
        source_id = f"resume_{profile_id}_{self._hash_content(content)}"

        logger.info(
            "Starting resume ingestion",
            profile_id=profile_id,
            filename=filename,
            source_id=source_id,
        )

        # Check if already ingested
        skip_result = self._check_skip(source_id, "resume")
        if skip_result:
            return skip_result

        try:
            # Parse resume
            parse_result = self.resume_parser.parse(content)
            logger.info(
                "Resume parsed successfully",
                sections=parse_result.metadata.get("detected_sections"),
            )

            # Prepare metadata
            metadata = parse_result.metadata.copy()
            metadata.update(
                {
                    "source_id": source_id,
                    "profile_id": profile_id,
                    "filename": filename,
                    "source_type": "resume",
                }
            )

            # Chunk the content
            chunks = self.strategy_selector.chunk(parse_result.text, metadata)
            logger.info("Resume chunked successfully", chunk_count=len(chunks))

            # Generate embeddings and store
            self.batch_processor.process(chunks)
            logger.info("Resume embeddings stored", chunk_count=len(chunks))

            # Record in database
            self._record_ingested_source(source_id, "resume", profile_id, len(chunks))

            return IngestResult(
                source_id=source_id,
                chunk_count=len(chunks),
                skipped=False,
            )

        except Exception as e:
            logger.error(
                "Resume ingestion failed",
                profile_id=profile_id,
                filename=filename,
                error=str(e),
            )
            raise

    def ingest_readme(
        self,
        profile_id: str,
        repo_name: str,
        content: str | bytes,
    ) -> IngestResult:
        """
        Ingest a README document.

        Args:
            profile_id: ID of the profile owner
            repo_name: Name of the repository
            content: README content (markdown string or bytes)

        Returns:
            IngestResult with ingestion status
        """
        source_id = f"readme_{profile_id}_{repo_name}_{self._hash_content(content)}"

        logger.info(
            "Starting README ingestion",
            profile_id=profile_id,
            repo_name=repo_name,
            source_id=source_id,
        )

        # Check if already ingested
        skip_result = self._check_skip(source_id, "readme")
        if skip_result:
            return skip_result

        try:
            # Parse README
            parse_result = self.readme_parser.parse(content)
            logger.info(
                "README parsed successfully",
                heading_count=parse_result.metadata.get("heading_count"),
                word_count=parse_result.metadata.get("word_count"),
            )

            # Prepare metadata
            metadata = parse_result.metadata.copy()
            metadata.update(
                {
                    "source_id": source_id,
                    "profile_id": profile_id,
                    "repo_name": repo_name,
                    "source_type": "readme",
                }
            )

            # Chunk the content
            chunks = self.strategy_selector.chunk(parse_result.text, metadata)
            logger.info("README chunked successfully", chunk_count=len(chunks))

            # Generate embeddings and store
            self.batch_processor.process(chunks)
            logger.info("README embeddings stored", chunk_count=len(chunks))

            # Record in database
            self._record_ingested_source(source_id, "readme", profile_id, len(chunks))

            return IngestResult(
                source_id=source_id,
                chunk_count=len(chunks),
                skipped=False,
            )

        except Exception as e:
            logger.error(
                "README ingestion failed",
                profile_id=profile_id,
                repo_name=repo_name,
                error=str(e),
            )
            raise

    def ingest_repo_metadata(
        self,
        profile_id: str,
        repo_data: dict,
    ) -> IngestResult:
        """
        Ingest repository metadata.

        Args:
            profile_id: ID of the profile owner
            repo_data: Repository metadata dictionary

        Returns:
            IngestResult with ingestion status
        """
        repo_name = repo_data.get("name", "unknown")
        source_id = f"repo_{profile_id}_{repo_name}_{self._hash_content(str(repo_data))}"

        logger.info(
            "Starting repo metadata ingestion",
            profile_id=profile_id,
            repo_name=repo_name,
            source_id=source_id,
        )

        # Check if already ingested
        skip_result = self._check_skip(source_id, "repo")
        if skip_result:
            return skip_result

        try:
            # Analyze repository
            parse_result = self.repo_analyzer.parse(repo_data)
            logger.info(
                "Repository analyzed successfully",
                language=parse_result.metadata.get("primary_language"),
                tech_stack=parse_result.metadata.get("tech_stack"),
            )

            # Prepare metadata
            metadata = parse_result.metadata.copy()
            metadata.update(
                {
                    "source_id": source_id,
                    "profile_id": profile_id,
                    "source_type": "repo",
                }
            )

            # Chunk the content
            chunks = self.strategy_selector.chunk(parse_result.text, metadata)
            logger.info("Repository metadata chunked successfully", chunk_count=len(chunks))

            # Generate embeddings and store
            self.batch_processor.process(chunks)
            logger.info("Repository embeddings stored", chunk_count=len(chunks))

            # Record in database
            self._record_ingested_source(source_id, "repo", profile_id, len(chunks))

            return IngestResult(
                source_id=source_id,
                chunk_count=len(chunks),
                skipped=False,
            )

        except Exception as e:
            logger.error(
                "Repository ingestion failed",
                profile_id=profile_id,
                repo_name=repo_name,
                error=str(e),
            )
            raise

    def ingest_portfolio(
        self,
        profile_id: str,
        url: str,
    ) -> IngestResult:
        """
        Fetch and ingest a portfolio web page.

        Args:
            profile_id: ID of the profile owner
            url: Portfolio page URL (http or https)

        Returns:
            IngestResult with ingestion status

        Raises:
            ValueError: If the URL is refused or the page has no extractable text
        """
        logger.info("Starting portfolio ingestion", profile_id=profile_id, url=url)

        try:
            html = self._fetch_page(url)
            source_id = f"portfolio_{profile_id}_{self._hash_content(html)}"

            # Check if already ingested
            skip_result = self._check_skip(source_id, "portfolio")
            if skip_result:
                return skip_result

            # Parse page
            parse_result = self.web_parser.parse(html)
            logger.info(
                "Portfolio parsed successfully",
                heading_count=parse_result.metadata.get("heading_count"),
                word_count=parse_result.metadata.get("word_count"),
            )

            # Prepare metadata
            metadata = parse_result.metadata.copy()
            metadata.update(
                {
                    "source_id": source_id,
                    "profile_id": profile_id,
                    "url": url,
                    "source_type": "portfolio",
                }
            )

            # Chunk the content
            chunks = self.strategy_selector.chunk(parse_result.text, metadata)
            logger.info("Portfolio chunked successfully", chunk_count=len(chunks))

            # Generate embeddings and store
            self.batch_processor.process(chunks)
            logger.info("Portfolio embeddings stored", chunk_count=len(chunks))

            # Record in database
            self._record_ingested_source(source_id, "portfolio", profile_id, len(chunks))

            return IngestResult(
                source_id=source_id,
                chunk_count=len(chunks),
                skipped=False,
            )

        except Exception as e:
            logger.error(
                "Portfolio ingestion failed",
                profile_id=profile_id,
                url=url,
                error=str(e),
            )
            raise

    def _fetch_page(self, url: str) -> str:
        """
        Fetch an HTML page, refusing URLs that point at internal addresses.

        Redirects are followed by hand (at most MAX_REDIRECTS) so every hop is checked.

        Raises:
            ValueError: If the URL, a redirect, the content type, or the size is refused
        """
        with httpx.Client(
            timeout=FETCH_TIMEOUT_SECONDS,
            follow_redirects=False,
            transport=self.http_transport,
        ) as client:
            for _ in range(MAX_REDIRECTS + 1):
                self._check_url_allowed(url)

                with client.stream("GET", url) as response:
                    if response.is_redirect:
                        location = response.headers.get("location")
                        if not location:
                            raise ValueError("Redirect without a Location header")
                        url = str(response.url.join(location))
                        continue

                    response.raise_for_status()

                    content_type = response.headers.get("content-type", "")
                    if "text/html" not in content_type.lower():
                        raise ValueError(f"Not an HTML page: {content_type or 'no content type'}")

                    body = bytearray()
                    for data in response.iter_bytes():
                        body.extend(data)
                        if len(body) > MAX_PAGE_BYTES:
                            raise ValueError("Page is larger than the size limit")

                    return body.decode(response.encoding or "utf-8", errors="replace")

        raise ValueError("Too many redirects")

    def _check_url_allowed(self, url: str) -> None:
        """
        Refuse non-http(s) URLs and hosts that resolve to a non-public IP address.

        Raises:
            ValueError: If the URL is not allowed
        """
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https"):
            raise ValueError(f"URL scheme not allowed: {parts.scheme or 'none'}")
        if not parts.hostname:
            raise ValueError("URL has no hostname")

        try:
            port = parts.port or (443 if parts.scheme == "https" else 80)
            addresses = socket.getaddrinfo(parts.hostname, port, proto=socket.IPPROTO_TCP)
        except (socket.gaierror, ValueError) as e:
            raise ValueError(f"Could not resolve host: {parts.hostname}") from e

        for address in addresses:
            ip = ipaddress.ip_address(str(address[4][0]).split("%")[0])
            if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
                ip = ip.ipv4_mapped
            if not ip.is_global or ip.is_multicast:
                raise ValueError(f"URL resolves to a non-public address: {parts.hostname}")

    def _hash_content(self, content: str | bytes) -> str:
        """Generate a hash of content for deduplication."""
        if isinstance(content, str):
            content = content.encode()
        return hashlib.sha256(content).hexdigest()[:16]

    def _check_skip(self, source_id: str, source_type: str) -> IngestResult | None:
        """
        Check if source has already been ingested.

        Returns IngestResult if should skip, None if should proceed.
        """
        try:
            # Query database for existing source
            # This assumes a table/model named IngestedSource
            existing = (
                self.db_session.query("IngestedSource")  # Placeholder - actual query depends on ORM
                .filter_by(source_id=source_id)
                .first()
            )

            if existing:
                logger.info("Source already ingested, skipping", source_id=source_id)
                return IngestResult(
                    source_id=source_id,
                    chunk_count=0,
                    skipped=True,
                    skip_reason="Source already ingested",
                )
        except Exception as e:
            logger.warning(
                "Could not check if source already ingested",
                source_id=source_id,
                error=str(e),
            )

        return None

    def _record_ingested_source(
        self,
        source_id: str,
        source_type: str,
        profile_id: str,
        chunk_count: int,
    ) -> None:
        """
        Record that a source has been ingested.

        Args:
            source_id: Unique ID for the source
            source_type: Type of source (resume, readme, repo)
            profile_id: ID of profile owner
            chunk_count: Number of chunks created
        """
        try:
            # This is a placeholder for actual database recording
            # In a real implementation, would create IngestedSource record
            logger.info(
                "Recording ingested source",
                source_id=source_id,
                source_type=source_type,
                profile_id=profile_id,
                chunk_count=chunk_count,
            )
        except Exception as e:
            logger.error(
                "Failed to record ingested source",
                source_id=source_id,
                error=str(e),
            )
