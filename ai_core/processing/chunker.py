import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from collections import Counter
from typing import TypedDict, List



# =========================================================
# CHUNK OBJECT SCHEMA
# =========================================================

class ChunkData(TypedDict):

    # Main chunk text
    text: str

    # Document metadata
    file_name: str
    document_title: str
    doc_id: str

    # Structural metadata
    section: str
    subsection: str

    # NEW
    chunk_type: str

    # NEW
    hierarchy_level: int

    # Chunk tracking
    chunk_id: int

    # Chunk statistics
    chunk_length: int
    word_count: int

    # Semantic metadata
    keywords: List[str]
    tags: List[str]

    # Table metadata
    sheet_name: str
    row_range: str
    column_names: List[str]


# =========================================================
# REMOVE REPEATED HEADERS / FOOTERS
# =========================================================

def remove_repeated_lines(text: str) -> str:
    """
    Removes repeated PDF headers and footers.

    Why this matters:
    -----------------
    PDFs often repeat:
    - document title
    - page headers
    - page footers
    - page numbers

    on EVERY page.

    These repeated lines pollute:
    - embeddings
    - retrieval quality
    - reranker scores

    Strategy:
    ---------
    1. Split document into lines
    2. Count frequency of each line
    3. Remove lines repeated too many times
    """

    # Split text into individual lines
    lines = text.splitlines()

    # Normalize lines for accurate counting
    normalized_lines = [
        line.strip()
        for line in lines
        if line.strip()
    ]

    # Count occurrences of each line
    line_counts = Counter(normalized_lines)

    cleaned_lines = []

    for line in lines:

        stripped = line.strip()

        # Skip empty lines
        if not stripped:
            cleaned_lines.append(line)
            continue

        # Remove lines repeated many times
        # Example:
        # "Task 5 -.NET Core Web API Report"
        if line_counts[stripped] > 3:
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


# =========================================================
# CLEAN RAW PDF TEXT
# =========================================================

def clean_text(text: str) -> str:
    """
    Advanced PDF cleanup for RAG systems.

    Fixes:
    -------
    - repeated headers
    - broken spacing
    - flattened headings
    - PDF artifacts
    - bad line breaks
    """

    # -----------------------------------------------------
    # STEP 1: Remove repeated lines
    # -----------------------------------------------------

    text = remove_repeated_lines(text)
    # REMOVE REPEATED REPORT TITLES
    text = re.sub(
        r"Task\s*\d+\s*-\s*\.NET\s*Core\s*Web\s*API\s*Report",
        " ",
        text,
        flags=re.IGNORECASE
    )
    # REMOVE REPEATED REPORT TITLES

    text = re.sub(
        r"Task\s*\d+\s*-\s*\.NET\s*Core\s*Web\s*API\s*Report",
        " ",
        text,
        flags=re.IGNORECASE
    )

    # -----------------------------------------------------
    # STEP 2: Normalize whitespace
    # -----------------------------------------------------

    text = text.replace("\r", "\n")

    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text)

    # Restore paragraph spacing
    text = re.sub(
        r"(\d+\.)",
        r"\n\n\1",
        text
    )

    # -----------------------------------------------------
    # STEP 3: Add line breaks BEFORE section headings
    #
    # Example:
    # "1. Objective"
    # "2. Technology Stack"
    # -----------------------------------------------------

    text = re.sub(
        r"(\s)(\d+\.\s+[A-Z])",
        r"\n\n\2",
        text
    )

    # -----------------------------------------------------
    # STEP 4: Add line breaks before common headings
    # -----------------------------------------------------

    headings = [
        "Technology Stack",
        "Implementation Details",
        "Evidence of Success",
        "Challenges & Solutions",
        "Key Learnings",
        "Swagger UI",
        "Endpoints Created"
    ]

    for heading in headings:

        text = text.replace(
            heading,
            f"\n\n{heading}"
        )

    # -----------------------------------------------------
    # STEP 5: Normalize blank lines
    # -----------------------------------------------------

    text = re.sub(r"\n{3,}", "\n\n", text)

    # -----------------------------------------------------
    # STEP 6: Fix punctuation spacing
    # -----------------------------------------------------

    text = re.sub(
        r"\s+([.,;:!?])",
        r"\1",
        text
    )

    # -----------------------------------------------------
    # STEP 7: Remove isolated page numbers
    # -----------------------------------------------------

    text = re.sub(r"\n\d+\n", "\n", text)

    return text.strip()


# =========================================================
# SPLIT DOCUMENT INTO SEMANTIC SECTIONS
# =========================================================

def split_by_headings(text: str):
    """
    Split document into semantic sections.
    Supports:
    - 1. Heading
    - 1.1 Subheading
    - ALL CAPS headings
    - Markdown headings
    """

    pattern = (
        r"(?="
        r"\n\s*(?:"
        r"\d+\.\s+[A-Z]|"          # 1. Heading
        r"\d+\.\d+\s+[A-Z]|"       # 1.1 Subheading
        r"#+\s+.*|"                # Markdown
        r"[A-Z][A-Z\s]{5,}"        # ALL CAPS
        r")"
        r")"
    )

    sections = re.split(pattern, text)

    sections = [
        s.strip()
        for s in sections
        if len(s.strip()) > 100
    ]

    return sections


# =========================================================
# SPLIT TABULAR DATA
# =========================================================

def split_tabular_data(text: str) -> list[str]:
    """
    Split CSV / Excel text into row-based chunks.

    Strategy:
    ----------
    - Each row becomes one semantic unit
    - Preserve structured field relationships
    - Better for retrieval accuracy
    """

    sections = []

    # Split by double newlines
    rows = text.split("\n\n")

    for row in rows:

        cleaned = row.strip()

        # Skip weak rows
        if len(cleaned) < 50:
            continue

        sections.append(cleaned)

    print(f"[TABULAR] Rows detected: {len(sections)}")

    return sections


# =========================================================
# DETECT CHUNK TYPE
# =========================================================

def detect_chunk_type(text: str) -> str:
    """
    Detect semantic chunk category.
    """

    lower = text.lower()

    # =====================================================
    # TABLE DETECTION
    # =====================================================

    if is_table_chunk(text):
        return "table"

    # =====================================================
    # CODE DETECTION
    # =====================================================

    code_patterns = [
        "def ",
        "class ",
        "import ",
        "public ",
        "private ",
        "SELECT ",
        "{",
        "}"
    ]

    if any(p in text for p in code_patterns):
        return "code"

    # =====================================================
    # LIST DETECTION
    # =====================================================

    list_patterns = [
        "- ",
        "* ",
        "1.",
        "2."
    ]

    if any(p in text for p in list_patterns):
        return "list"

    # =====================================================
    # DEFAULT
    # =====================================================

    return "paragraph"

# =========================================================
# DETECT HEADING LEVEL
# =========================================================

def detect_heading_level(section_title: str) -> int:
    """
    Detect heading hierarchy level.
    """

    if re.match(r"\d+\.\d+\.\d+", section_title):
        return 3

    if re.match(r"\d+\.\d+", section_title):
        return 2

    if re.match(r"\d+\.", section_title):
        return 1

    return 0


# =========================================================
# DETECT TABLE-LIKE CONTENT
# =========================================================

def is_table_chunk(text: str) -> bool:
    """
    Detect whether chunk contains
    structured table-like data.
    """

    indicators = [

        # CSV style
        ",",

        # Excel converted rows
        "|",

        # Multiple colons
        ":",

        # Tabs
        "\t"
    ]

    score = 0

    for indicator in indicators:

        if indicator in text:
            score += 1

    return score >= 2



# =========================================================
# ADVANCED RAG CHUNKER
# =========================================================

def chunk_text(text: str,file_name: str,doc_id: str) -> List[ChunkData]:
    """
    Production-grade chunking pipeline.

    Features:
    ---------
    - PDF cleanup
    - Header/footer removal
    - Section awareness
    - Paragraph preservation
    - Sentence-aware splitting
    - Semantic chunk overlap
    """

    # -----------------------------------------------------
    # STEP 1: CLEAN RAW TEXT
    # -----------------------------------------------------

    text = clean_text(text)
    document_title = (
    file_name
    .replace(".pdf", "")
    .replace(".docx", "")
    .replace("_", " ")
    .strip()
    )

    # -----------------------------------------------------
    # STEP 2: SPLIT INTO SECTIONS
    # -----------------------------------------------------

    # DETECT TABULAR DATA
    is_tabular = (
        "Rig_ID:" in text
        or "Sheet:" in text
        or "===" in text
    )

    # SPLIT STRATEGY
    if is_tabular:

        print("[CHUNKER] Using TABULAR chunking")

        sections = split_tabular_data(text)

    else:

        print("[CHUNKER] Using DOCUMENT chunking")

        sections = split_by_headings(text)

    # -----------------------------------------------------
    # STEP 3: CONFIGURE SPLITTER
    # -----------------------------------------------------

    splitter = RecursiveCharacterTextSplitter(

        # Larger chunks preserve technical meaning
        chunk_size=500,

        # Overlap preserves cross-boundary context
        chunk_overlap=80,

        # Semantic-aware separators
        separators=[
            "\n# ",
            "\n## ",
            "\n### ",

            "\n\n",

            "\n",

            ". ",
            "? ",
            "! ",

            "; ",

            "- ",
            "* ",

            "```",

            "    ",

            " ",

            ""
        ]
    )

    all_chunks: List[ChunkData] = []

    # -----------------------------------------------------
    # STEP 4: PROCESS EACH SECTION
    # -----------------------------------------------------

    for section in sections:

        # Extract first line as section title
        lines = section.splitlines()

        # =====================================================
        # EXTRACT CLEAN SECTION TITLE
        # =====================================================

        section_title = "General"

        first_line = lines[0].strip()

        match = re.match(
            r"(\d+)\.\s*(.+)",
            first_line
        )

        if match:

            section_title = match.group(2).strip()

        # Split section into smaller chunks
        chunks = splitter.split_text(section)

        # -------------------------------------------------
        # Process each generated chunk
        # -------------------------------------------------

        for chunk in chunks:

            # chunk is plain STRING
            cleaned = chunk.strip()

            chunk_title = (cleaned[:80].replace("\n", " ").strip())

            # Skip low-information chunks
            if len(cleaned.split()) < 8:
                continue

            # Remove duplicated title
            cleaned = cleaned.replace(
                section_title,
                "",
                1
            ).strip()

            # =====================================================
            # TABLE-AWARE ENRICHMENT
            # =====================================================

            chunk_type = detect_chunk_type(cleaned)

            if chunk_type == "table":

                enhanced_chunk = f"""
                Document: {document_title}

                Section: {section_title}

                Content Type:
                Structured Table Data

                Table Content:
                {cleaned}
                """

            else:

                enhanced_chunk = f"""
                Document: {document_title}

                Section: {section_title}

                Chunk Title:
                {chunk_title}

                Content:
                {cleaned}
                """

            # Store structured chunk
            chunk_data: ChunkData = {

                # =====================================================
                # MAIN CONTENT
                # =====================================================

                "text": enhanced_chunk,

                # =====================================================
                # DOCUMENT METADATA
                # =====================================================

                "file_name": file_name,

                "document_title": document_title,

                "doc_id": doc_id,

                # =====================================================
                # STRUCTURE METADATA
                # =====================================================

                "section": section_title,

                "subsection": "General",

                # NEW
                "chunk_type": detect_chunk_type(cleaned),

                # NEW
                "hierarchy_level": detect_heading_level(
                    first_line
                ),

                # =====================================================
                # CHUNK TRACKING
                # =====================================================

                "chunk_id": len(all_chunks),

                # =====================================================
                # CHUNK STATS
                # =====================================================

                "chunk_length": len(enhanced_chunk),

                "word_count": len(enhanced_chunk.split()),

                # =====================================================
                # SEMANTIC METADATA
                # =====================================================

                "keywords": extract_keywords(cleaned),

                "tags": [],

                "sheet_name": "General",

                "row_range": "0-0",

                "column_names": [],
            }

            all_chunks.append(chunk_data)

    # -----------------------------------------------------
    # STEP 5: DEBUGGING
    # -----------------------------------------------------

    print("\n" + "=" * 80)
    print("[ADVANCED CHUNK DEBUG]")
    print("=" * 80)

    print(f"Total Chunks: {len(all_chunks)}")

    for i, chunk_obj in enumerate(all_chunks[:5]):

        chunk = chunk_obj["text"]

        print("\n" + "-" * 80)
        print(f"Chunk {i + 1}")
        print("-" * 80)

        print(f"Section     : {chunk_obj['section']}")
        print(f"Length      : {len(chunk)}")
        print(f"Word Count  : {len(chunk.split())}")

        print("\nPreview:\n")
        print(chunk[:500])

    all_chunks = deduplicate_chunks(all_chunks)

    return all_chunks

def deduplicate_chunks(chunks):

    seen = set()
    unique_chunks = []

    for chunk in chunks:

        normalized = (
            chunk["text"]
            .strip()
            .lower()
        )

        if normalized in seen:
            continue

        seen.add(normalized)
        unique_chunks.append(chunk)

    return unique_chunks

def extract_keywords(text: str, top_k=10):
    """
    Simple keyword extraction using word frequency.
    """

    words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())

    stopwords = {
        "the", "and", "is", "in", "to", "of",
        "a", "for", "on", "with", "that",
        "this", "as", "by", "an", "be",
        "are", "or", "from"
    }

    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())

    filtered = [
        w for w in words
        if w not in stopwords
    ]

    counts = Counter(filtered)

    return [
        word
        for word, _ in counts.most_common(top_k)
    ]


