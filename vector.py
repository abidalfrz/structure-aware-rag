import os
import re
import warnings
import pymupdf4llm
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

warnings.filterwarnings("ignore")

FILE_PATH = os.path.join(os.path.dirname(__file__), "data", "Buku Panduan SSDS 2026.pdf")

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_loc = "./chroma_db"
add_docs = not os.path.exists(db_loc)

def clean_markdown(md_text):
    noise_patterns = [
        r"SSF\s+Sebelas Maret Statistics Fair\s+202[56]",
        r"BUKU PANDUAN PENYISIHAN.*",
        r"BUKU PANDUAN SSDS.*"
    ]

    for pattern in noise_patterns:
        md_text = re.sub(pattern, "", md_text, flags=re.IGNORECASE | re.MULTILINE)

    md_text = re.sub(r'^\s*#{1,6}\s*\*+\s*$', '', md_text, flags=re.MULTILINE)

    md_text = re.sub(r'\*\*(.*?)\*\*\s+\1', r'**\1**', md_text)

    md_text = re.sub(
        r'D\.\s*PERSYARATAN DAN KETENTUAN\*+\s*\*+PESERTA',
        'D. PERSYARATAN DAN KETENTUAN PESERTA',
        md_text,
        flags=re.IGNORECASE
    )

    md_text = re.sub(
        r'^\s*#{0,6}\s*-?\s*\*{0,2}\s*([A-L])\.\s+(.+?)\s*\*{0,2}\s*$',
        r'# \1. \2',
        md_text,
        flags=re.MULTILINE
    )

    md_text = re.sub(
        r'^# K\.\s*NARAHUBUNG\s+(.+)$',
        r'# K. NARAHUBUNG\n\1',
        md_text,
        flags=re.MULTILINE | re.IGNORECASE
    )

    md_text = re.sub(
        r'^# L\.\s*KONTAK SSF\s+(.+)$',
        r'# L. KONTAK SSF\n\1',
        md_text,
        flags=re.MULTILINE | re.IGNORECASE
    )

    roman_subheadings = [
        "Gambaran Umum Lomba",
        "Teknis Babak Penyisihan",
        "Teknis Babak Final"
    ]

    for title in roman_subheadings:
        md_text = re.sub(
            rf'^\s*#{0,6}\s*\*{{0,2}}\s*(I{{1,3}})\.\s*{re.escape(title)}\s*\*{{0,2}}\s*$',
            rf'## \1. {title}',
            md_text,
            flags=re.MULTILINE | re.IGNORECASE
        )

    md_text = re.sub(r'[ \t]+\n', '\n', md_text)
    md_text = re.sub(r'\n{3,}', '\n\n', md_text)

    return md_text

def normalize_markdown_table(text):
    lines = text.splitlines()
    result = []

    for line in lines:
        stripped = line.strip()

        if not stripped.startswith("|"):
            result.append(line)
            continue

        cells = [cell.strip() for cell in stripped.strip("|").split("|")]

        if len(cells) != 2:
            continue

        left = re.sub(r'[*_]', '', cells[0]).strip()
        right = re.sub(r'[*_]', '', cells[1]).strip()

        if left.replace("-", "").strip() == "":
            continue

        if left.lower() == "kegiatan" and right.lower() == "waktu pelaksanaan":
            continue

        result.append(f"{left}: {right}")

    return "\n".join(result)


def load_documents():

    md_text = pymupdf4llm.to_markdown(FILE_PATH)
    md_text = clean_markdown(md_text)

    with open("cleaned_document.md", "w", encoding="utf-8") as f:
        f.write(md_text)

    headers_to_split_on = [
        ("#", "Bab Utama"),
        ("##", "Sub Bab")
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False
    )

    header_splits = markdown_splitter.split_text(md_text)
    print("Header splits: ")
    for i, split in enumerate(header_splits):
        bab_utama = split.metadata.get("Bab Utama", "N/A")
        sub_bab = split.metadata.get("Sub Bab", "N/A")
        print(f"Split {i+1} | Bab: {bab_utama} | Sub Bab: {sub_bab} | Preview: {split.page_content[:80]}...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1800,
        chunk_overlap=250,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    documents = []

    for split in header_splits:
        chunks = text_splitter.split_text(split.page_content)

        for chunk in chunks:
            chunk = normalize_markdown_table(chunk)
            meta = split.metadata.copy()
            meta["source"] = FILE_PATH

            bab_utama = meta.get("Bab Utama", "")
            sub_bab = meta.get("Sub Bab", "")

            enriched_content = ""

            if bab_utama:
                enriched_content += f"Bab: {bab_utama}\n"

            if sub_bab:
                enriched_content += f"Sub Bab: {sub_bab}\n"

            enriched_content += chunk

            documents.append(
                Document(
                    page_content=enriched_content,
                    metadata=meta
                )
            )

    documents = filter_complex_metadata(documents)

    return documents


documents = load_documents()
if add_docs:

    print(f"\nTotal chunks: {len(documents)}")

    for i, doc in enumerate(documents):
        print("\n" + "=" * 80)
        print(f"Chunk {i+1}")
        print("Metadata:", doc.metadata)
        print(doc.page_content)


vector_store = Chroma(
    collection_name="ssds_data_mining",
    persist_directory=db_loc,
    embedding_function=embeddings
)


if add_docs:
    ids = [f"ssds_chunk_{i}" for i in range(len(documents))]

    vector_store.add_documents(
        documents=documents,
        ids=ids
    )


dense_retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

def bm25_tokenize(text):
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[_*#|:/\\.,!?()\[\]{}<>–—-]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.split()

bm25_retriever = BM25Retriever.from_documents(documents, preprocess_func=bm25_tokenize)
bm25_retriever.k = 3

retriever = EnsembleRetriever(
    retrievers=[dense_retriever, bm25_retriever],
    weights=[0.5, 0.5]
)

def debug_retriever(query, k=5):
    print(f"\nDebugging retriever for query: '{query}'")
    retrieved_docs = retriever.invoke(query)

    print(f"\nRetrieved {len(retrieved_docs)} documents:")
    for i, doc in enumerate(retrieved_docs[:k]):
        bab_utama = doc.metadata.get('Bab Utama', 'N/A')
        print(f"Chunk {i+1} | Bab: {bab_utama} | Preview: {doc.page_content.replace(chr(10), ' ')[:80]}...")

if __name__ == "__main__":
    query = "Tanggal berapa perilisan dataset?"
    debug_retriever(query, k=5)