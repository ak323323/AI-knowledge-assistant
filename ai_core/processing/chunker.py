import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from collections import Counter
from typing import TypedDict, List



# =========================================================
# CHUNK OBJECT SCHEMA
# =========================================================

class ChunkData(TypedDict):
    """
    Represents a semantic chunk used in RAG retrieval.

    Why TypedDict?
    ---------------
    - Gives strong typing
    - Fixes Pylance errors
    - Improves autocomplete
    - Prevents key mistakes
    """

    # Actual chunk text used for embeddings
    text: str

    # Section heading
    section: str

    # Chunk size metadata
    chunk_length: int


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

    Detects headings like:

    1. Objective
    2. Technology Stack
    3. Implementation Details
    """

    # -------------------------------------------------
    # Split BEFORE numbered headings
    # -------------------------------------------------

    pattern = r"(?=\n\s*\d+\.\s+[A-Z])"

    sections = re.split(pattern, text)

    # Cleanup
    sections = [
        s.strip()
        for s in sections
        if len(s.strip()) > 100
    ]

    return sections



# =========================================================
# ADVANCED RAG CHUNKER
# =========================================================

def chunk_text(text: str) -> List[ChunkData]:
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

    # -----------------------------------------------------
    # STEP 2: SPLIT INTO SECTIONS
    # -----------------------------------------------------

    sections = split_by_headings(text)

    # -----------------------------------------------------
    # STEP 3: CONFIGURE SPLITTER
    # -----------------------------------------------------

    splitter = RecursiveCharacterTextSplitter(

        # Larger chunks preserve technical meaning
        chunk_size=450,

        # Overlap preserves cross-boundary context
        chunk_overlap=80,

        # Semantic-aware separators
        separators=[

            # Large paragraph gaps
            "\n\n\n",

            # Paragraphs
            "\n\n",

            # Lines
            "\n",

            # Sentences
            ". ",
            "? ",
            "! ",

            # Bullet lists
            "- ",
            "* ",

            # Code blocks
            "```",

            # Indentation
            "    ",

            # Words
            " ",

            # Last fallback
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

            # Skip tiny chunks
            if len(cleaned) < 80:
                continue

            # Skip low-information chunks
            if len(cleaned.split()) < 10:
                continue

            # Remove duplicated title
            cleaned = cleaned.replace(
                section_title,
                "",
                1
            ).strip()

            enhanced_chunk = f"""
            Section: {section_title}

            {cleaned}
            """

            # Store structured chunk
            chunk_data: ChunkData = {

                                        "text": enhanced_chunk,

                                        "section": section_title,

                                        "chunk_length": len(enhanced_chunk)
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

    return all_chunks