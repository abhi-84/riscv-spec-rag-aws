"""
ingest.py - Build the vector store from PDF documents (RISC-V specs).

Changes vs. the original Insurellm POC ingest:
  1. Reads PDFs instead of markdown. Docling converts each PDF to markdown, page by page.
  2. Converted markdown is cached in CONVERTED_PATH, so PDFs are parsed only once.
  3. Each PDF is split into sections of PAGES_PER_SECTION pages before LLM chunking
     (a 100+ page document cannot be rewritten into chunks in a single LLM reply).
  4. Every chunk keeps metadata: source file, doc type (sub-folder), page range.
  5. Embeddings and Chroma inserts run in batches (API request-size limits).
Unchanged: LLM-based chunking with headline/summary/original text, OpenAI embeddings, Chroma.

Folder layout expected (folder name = doc "type"):
  knowledge-base-riscv/
      profiles/   rva20.pdf, rva22.pdf, rva23.pdf
      sbi/        riscv-sbi.pdf
      boot/       riscv-brs.pdf
      vector/     riscv-v-spec.pdf
"""

import json
from pathlib import Path
from multiprocessing import Pool

from chromadb import PersistentClient
from dotenv import load_dotenv
from litellm import completion
from openai import OpenAI
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

load_dotenv(override=True)

# ---------------------------------------------------------------- configuration
MODEL = "openai/gpt-4o-mini"                      # chunking model (unchanged)
EMBEDDING_MODEL = "text-embedding-3-large"        # unchanged; must match answer.py
ROOT = Path(__file__).parent.parent
KNOWLEDGE_BASE_PATH = ROOT / "my_rag_riscv/knowledge-base-riscv"   # input PDFs
CONVERTED_PATH = ROOT / "converted-md"                 # cached Docling output (one JSON per PDF)
DB_NAME = str(ROOT / "preprocessed_db")
COLLECTION_NAME = "docs"

PAGES_PER_SECTION = 6        # pages sent to the LLM per chunking call
AVERAGE_CHUNK_SIZE = 1000    # characters; drives the "how many chunks" hint in the prompt
MIN_SECTION_CHARS = 200      # skip near-empty sections (cover pages, blank pages)
EMBED_BATCH = 100            # texts per embeddings request
WORKERS = 3                  # set to 1 if you hit rate limits

wait = wait_exponential(multiplier=1, min=10, max=240)
openai = OpenAI()


# ---------------------------------------------------------------- data models
class Chunk(BaseModel):
    headline: str = Field(
        description="A brief heading for this chunk, typically a few words, that is most likely to be surfaced in a query",
    )
    summary: str = Field(
        description="A few sentences summarizing the content of this chunk to answer common questions"
    )
    original_text: str = Field(
        description="The original text of this chunk from the provided document, exactly as is, not changed in any way"
    )


class Chunks(BaseModel):
    chunks: list[Chunk]


# ---------------------------------------------------------------- 1. PDF -> markdown (per page)
def convert_pdf(pdf_path: Path) -> list[str]:
    """Return a list of markdown strings, one per page. Cached as JSON next to other conversions."""
    cache_file = CONVERTED_PATH / (pdf_path.relative_to(KNOWLEDGE_BASE_PATH).as_posix().replace("/", "__") + ".json")
    if cache_file.exists() and cache_file.stat().st_mtime >= pdf_path.stat().st_mtime:
        return json.loads(cache_file.read_text(encoding="utf-8"))

    from docling.document_converter import DocumentConverter  # imported lazily: slow first import

    doc = DocumentConverter().convert(str(pdf_path)).document
    pages = [doc.export_to_markdown(page_no=p) for p in range(1, doc.num_pages() + 1)]

    CONVERTED_PATH.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(pages, ensure_ascii=False), encoding="utf-8")
    return pages


# ---------------------------------------------------------------- 2. pages -> sections
def split_into_sections(pages: list[str], pages_per_section: int = PAGES_PER_SECTION) -> list[dict]:
    """Group consecutive pages into sections and record their page range (1-based)."""
    sections = []
    for start in range(0, len(pages), pages_per_section):
        group = pages[start:start + pages_per_section]
        text = "\n\n".join(p for p in group if p.strip())
        if len(text) < MIN_SECTION_CHARS:
            continue
        sections.append({
            "text": text,
            "page_start": start + 1,
            "page_end": start + len(group),
        })
    return sections


def fetch_documents() -> list[dict]:
    """Convert every PDF and return one dict per section, in the format the rest of the script uses."""
    documents = []
    pdfs = sorted(KNOWLEDGE_BASE_PATH.rglob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs found under {KNOWLEDGE_BASE_PATH}")

    for pdf in tqdm(pdfs, desc="Converting PDFs"):
        doc_type = pdf.parent.name if pdf.parent != KNOWLEDGE_BASE_PATH else "general"
        pages = convert_pdf(pdf)
        for section in split_into_sections(pages):
            documents.append({
                "type": doc_type,
                "source": pdf.relative_to(ROOT).as_posix(),
                "page_start": section["page_start"],
                "page_end": section["page_end"],
                "text": section["text"],
            })

    print(f"Loaded {len(pdfs)} PDFs -> {len(documents)} sections")
    return documents


# ---------------------------------------------------------------- 3. LLM chunking (original approach)
def make_prompt(document: dict) -> str:
    how_many = (len(document["text"]) // AVERAGE_CHUNK_SIZE) + 1
    return f"""
You take a section of a document and you split it into overlapping chunks for a Knowledge Base.

The document is a RISC-V technical specification.
The document is of type: {document["type"]}
The document has been retrieved from: {document["source"]}
This section covers pages {document["page_start"]} to {document["page_end"]}.

A chatbot will use these chunks to answer technical questions about RISC-V.
You should divide up the section as you see fit, being sure that the entire section is returned across the chunks - don't leave anything out.
This section should probably be split into at least {how_many} chunks, but you can have more or less as appropriate, ensuring that there are individual chunks to answer specific questions.
Keep tables, register names, instruction mnemonics and encodings intact; never split a table row across chunks.
There should be overlap between the chunks as appropriate; typically about 25% overlap or about 50 words, so you have the same text in multiple chunks for best retrieval results.

For each chunk, you should provide a headline, a summary, and the original text of the chunk.
Together your chunks should represent the entire section with overlap.

Here is the section:

{document["text"]}

Respond with the chunks.
"""


def make_messages(document: dict) -> list[dict]:
    return [{"role": "user", "content": make_prompt(document)}]


@retry(wait=wait, stop=stop_after_attempt(3))
def process_document(document: dict) -> list[dict]:
    response = completion(model=MODEL, messages=make_messages(document), response_format=Chunks)
    reply = response.choices[0].message.content
    doc_as_chunks = Chunks.model_validate_json(reply).chunks
    pages = f"{document['page_start']}-{document['page_end']}"
    return [
        {
            "page_content": f"{c.headline}\n\n{c.summary}\n\n{c.original_text}",
            "metadata": {
                "source": document["source"],
                "type": document["type"],
                "pages": pages,
                "page_start": document["page_start"],
                "page_end": document["page_end"],
            },
        }
        for c in doc_as_chunks
    ]


def create_chunks(documents: list[dict]) -> list[dict]:
    """Chunk sections in parallel. If you get a rate limit error, set WORKERS to 1."""
    chunks = []
    with Pool(processes=WORKERS) as pool:
        for result in tqdm(pool.imap_unordered(process_document, documents), total=len(documents), desc="Chunking"):
            chunks.extend(result)
    return chunks


# ---------------------------------------------------------------- 4. embeddings + Chroma
@retry(wait=wait, stop=stop_after_attempt(3))
def embed_batch(texts: list[str]) -> list[list[float]]:
    data = openai.embeddings.create(model=EMBEDDING_MODEL, input=texts).data
    return [e.embedding for e in data]


def create_embeddings(chunks: list[dict]) -> None:
    chroma = PersistentClient(path=DB_NAME)
    if COLLECTION_NAME in [c.name for c in chroma.list_collections()]:
        chroma.delete_collection(COLLECTION_NAME)
    collection = chroma.get_or_create_collection(COLLECTION_NAME)

    for start in tqdm(range(0, len(chunks), EMBED_BATCH), desc="Embedding"):
        batch = chunks[start:start + EMBED_BATCH]
        texts = [c["page_content"] for c in batch]
        collection.add(
            ids=[str(start + i) for i in range(len(batch))],
            embeddings=embed_batch(texts),
            documents=texts,
            metadatas=[c["metadata"] for c in batch],
        )
    print(f"Vectorstore created with {collection.count()} chunks")


if __name__ == "__main__":
    documents = fetch_documents()
    chunks = create_chunks(documents)
    create_embeddings(chunks)
    print("Ingestion complete")
