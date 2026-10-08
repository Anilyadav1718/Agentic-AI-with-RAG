import hashlib
import re

import chromadb

from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from sentence_transformers import (
    SentenceTransformer,
)


CHROMA_PATH = "chroma_db"

COLLECTION_NAME = (
    "nexor_ai_documents"
)

EMBEDDING_MODEL = (
    "all-MiniLM-L6-v2"
)

FINAL_TOP_K = 3

SEMANTIC_CANDIDATES = 10
KEYWORD_CANDIDATES = 10

RRF_K = 60

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150


embedding_model = (
    SentenceTransformer(
        EMBEDDING_MODEL
    )
)

chroma_client = (
    chromadb.PersistentClient(
        path=CHROMA_PATH
    )
)

collection = (
    chroma_client
    .get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine"
        },
    )
)


_active_document = None


def set_active_document(
    file_name,
):
    global _active_document
    _active_document = file_name


def get_active_document():
    return _active_document


def clear_active_document():
    global _active_document
    _active_document = None


def tokenize(text):
    if not text:
        return []

    return re.findall(
        r"\b\w+\b",
        text.lower(),
    )


def chunk_text(
    text,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP,
):
    if not text:
        return []

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(
            start + chunk_size,
            len(text),
        )

        chunk = text[
            start:end
        ].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(
            end - overlap,
            start + 1,
        )

    return chunks


def delete_document(file_name):
    try:
        collection.delete(
            where={
                "file": file_name
            }
        )

        return {
            "success": True
        }

    except Exception as error:
        return {
            "success": False,
            "error": str(error),
        }


def process_pdf(
    pdf_path,
    file_name,
):
    global _active_document

    try:
        reader = PdfReader(
            pdf_path
        )

        if not reader.pages:
            return {
                "success": False,
                "error": (
                    "PDF contains no "
                    "readable pages."
                ),
            }

        delete_document(
            file_name
        )

        documents = []
        metadatas = []
        ids = []

        readable_pages = 0
        chunk_count = 0

        for page_number, page in enumerate(
            reader.pages,
            start=1,
        ):
            try:
                text = (
                    page.extract_text()
                    or ""
                )
            except Exception:
                text = ""

            text = text.strip()

            if not text:
                continue

            readable_pages += 1

            chunks = chunk_text(
                text
            )

            for chunk_index, chunk in enumerate(
                chunks
            ):
                unique_string = (
                    f"{file_name}:"
                    f"{page_number}:"
                    f"{chunk_index}:"
                    f"{chunk}"
                )

                chunk_id = (
                    hashlib.sha256(
                        unique_string.encode(
                            "utf-8"
                        )
                    ).hexdigest()
                )

                documents.append(
                    chunk
                )

                metadatas.append(
                    {
                        "file": file_name,
                        "page": page_number,
                        "chunk": chunk_index,
                    }
                )

                ids.append(
                    chunk_id
                )

                chunk_count += 1

        if not documents:
            return {
                "success": False,
                "error": (
                    "No readable text was "
                    "found in the PDF."
                ),
            }

        embeddings = (
            embedding_model.encode(
                documents,
                normalize_embeddings=True,
            )
            .tolist()
        )

        collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        _active_document = (
            file_name
        )

        return {
            "success": True,
            "file": file_name,
            "pages": readable_pages,
            "chunks": chunk_count,
        }

    except Exception as error:
        return {
            "success": False,
            "error": str(error),
        }


def get_document_chunks(
    file_name,
):
    try:
        result = collection.get(
            where={
                "file": file_name
            },
            include=[
                "documents",
                "metadatas",
            ],
        )

        documents = (
            result.get(
                "documents",
                [],
            )
            or []
        )

        metadatas = (
            result.get(
                "metadatas",
                [],
            )
            or []
        )

        ids = (
            result.get(
                "ids",
                [],
            )
            or []
        )

        chunks = []

        for index, document in enumerate(
            documents
        ):
            metadata = (
                metadatas[index]
                if index < len(
                    metadatas
                )
                else {}
            )

            chunk_id = (
                ids[index]
                if index < len(ids)
                else str(index)
            )

            chunks.append(
                {
                    "id": chunk_id,
                    "text": document,
                    "file": (
                        metadata.get(
                            "file",
                            file_name,
                        )
                    ),
                    "page": (
                        metadata.get(
                            "page",
                            "?",
                        )
                    ),
                    "chunk": (
                        metadata.get(
                            "chunk",
                            index,
                        )
                    ),
                }
            )

        return chunks

    except Exception:
        return []


def semantic_search(
    query,
    file_name,
    top_k=SEMANTIC_CANDIDATES,
):
    chunks = get_document_chunks(
        file_name
    )

    if not chunks:
        return []

    top_k = min(
        top_k,
        len(chunks),
    )

    query_embedding = (
        embedding_model.encode(
            [query],
            normalize_embeddings=True,
        )
        .tolist()[0]
    )

    result = collection.query(
        query_embeddings=[
            query_embedding
        ],
        n_results=top_k,
        where={
            "file": file_name
        },
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    documents = (
        result.get(
            "documents",
            [[]],
        )[0]
        or []
    )

    metadatas = (
        result.get(
            "metadatas",
            [[]],
        )[0]
        or []
    )

    distances = (
        result.get(
            "distances",
            [[]],
        )[0]
        or []
    )

    ids = (
        result.get(
            "ids",
            [[]],
        )[0]
        or []
    )

    results = []

    for rank, document in enumerate(
        documents,
        start=1,
    ):
        index = rank - 1

        distance = (
            distances[index]
            if index < len(
                distances
            )
            else 1.0
        )

        similarity = max(
            0.0,
            min(
                1.0,
                1.0 - float(
                    distance
                ),
            ),
        )

        metadata = (
            metadatas[index]
            if index < len(
                metadatas
            )
            else {}
        )

        results.append(
            {
                "id": (
                    ids[index]
                    if index < len(ids)
                    else str(index)
                ),
                "text": document,
                "file": (
                    metadata.get(
                        "file",
                        file_name,
                    )
                ),
                "page": (
                    metadata.get(
                        "page",
                        "?",
                    )
                ),
                "chunk": (
                    metadata.get(
                        "chunk",
                        index,
                    )
                ),
                "semantic_rank": rank,
                "semantic_relevance": (
                    round(
                        similarity * 100,
                        2,
                    )
                ),
            }
        )

    return results


def keyword_search(
    query,
    file_name,
    top_k=KEYWORD_CANDIDATES,
):
    chunks = get_document_chunks(
        file_name
    )

    if not chunks:
        return []

    query_tokens = tokenize(
        query
    )

    if not query_tokens:
        return []

    tokenized_corpus = [
        tokenize(
            chunk["text"]
        )
        for chunk in chunks
    ]

    bm25 = BM25Okapi(
        tokenized_corpus
    )

    scores = bm25.get_scores(
        query_tokens
    )

    ranked_indexes = sorted(
        range(len(scores)),
        key=lambda index: (
            scores[index]
        ),
        reverse=True,
    )

    results = []

    for index in ranked_indexes:
        score = float(
            scores[index]
        )

        if score <= 0:
            continue

        chunk = chunks[index]

        results.append(
            {
                **chunk,
                "keyword_score": (
                    round(
                        score,
                        4,
                    )
                ),
            }
        )

        if len(results) >= top_k:
            break

    for rank, result in enumerate(
        results,
        start=1,
    ):
        result[
            "keyword_rank"
        ] = rank

    return results


def reciprocal_rank_fusion(
    semantic_results,
    keyword_results,
):
    fused = {}

    for rank, result in enumerate(
        semantic_results,
        start=1,
    ):
        chunk_id = result["id"]

        if chunk_id not in fused:
            fused[chunk_id] = {
                **result,
                "rrf_score": 0.0,
                "keyword_score": 0.0,
                "semantic_present": False,
                "keyword_present": False,
            }

        fused[chunk_id][
            "semantic_present"
        ] = True

        fused[chunk_id][
            "semantic_rank"
        ] = rank

        fused[chunk_id][
            "semantic_relevance"
        ] = result.get(
            "semantic_relevance",
            0,
        )

        fused[chunk_id][
            "rrf_score"
        ] += (
            1.0
            / (
                RRF_K + rank
            )
        )

    for rank, result in enumerate(
        keyword_results,
        start=1,
    ):
        chunk_id = result["id"]

        if chunk_id not in fused:
            fused[chunk_id] = {
                **result,
                "rrf_score": 0.0,
                "semantic_relevance": 0.0,
                "semantic_present": False,
                "keyword_present": False,
            }

        fused[chunk_id][
            "keyword_present"
        ] = True

        fused[chunk_id][
            "keyword_rank"
        ] = rank

        fused[chunk_id][
            "keyword_score"
        ] = result.get(
            "keyword_score",
            0,
        )

        fused[chunk_id][
            "rrf_score"
        ] += (
            1.0
            / (
                RRF_K + rank
            )
        )

    ranked = sorted(
        fused.values(),
        key=lambda item: (
            item["rrf_score"]
        ),
        reverse=True,
    )

    final_results = ranked[
        :FINAL_TOP_K
    ]

    if not final_results:
        return []

    max_rrf = max(
        item["rrf_score"]
        for item in final_results
    )

    for result in final_results:
        semantic_present = (
            result.get(
                "semantic_present",
                False,
            )
        )

        keyword_present = (
            result.get(
                "keyword_present",
                False,
            )
        )

        if (
            semantic_present
            and keyword_present
        ):
            match_type = (
                "Semantic + Keyword"
            )

        elif semantic_present:
            match_type = "Semantic"

        else:
            match_type = "Keyword"

        result[
            "match_type"
        ] = match_type

        result[
            "relevance"
        ] = round(
            (
                result["rrf_score"]
                / max_rrf
            )
            * 100,
            2,
        )

    return final_results


def hybrid_search(
    query,
    file_name,
):
    semantic_results = (
        semantic_search(
            query,
            file_name,
        )
    )

    keyword_results = (
        keyword_search(
            query,
            file_name,
        )
    )

    return reciprocal_rank_fusion(
        semantic_results,
        keyword_results,
    )


def knowledge_base_search(
    query: str,
):
    file_name = (
        get_active_document()
    )

    if not file_name:
        return {
            "success": False,
            "error": (
                "No PDF is active. "
                "Upload a PDF first."
            ),
            "sources": [],
        }

    if not query or not query.strip():
        return {
            "success": False,
            "error": (
                "Knowledge base query "
                "cannot be empty."
            ),
            "sources": [],
        }

    try:
        results = hybrid_search(
            query.strip(),
            file_name,
        )

        sources = []

        for result in results:
            sources.append(
                {
                    "file": result.get(
                        "file",
                        file_name,
                    ),
                    "page": result.get(
                        "page",
                        "?",
                    ),
                    "text": result.get(
                        "text",
                        "",
                    ),
                    "relevance": (
                        result.get(
                            "relevance",
                            0,
                        )
                    ),
                    "semantic_relevance": (
                        result.get(
                            "semantic_relevance",
                            0,
                        )
                    ),
                    "keyword_score": (
                        result.get(
                            "keyword_score",
                            0,
                        )
                    ),
                    "match_type": (
                        result.get(
                            "match_type",
                            "Hybrid",
                        )
                    ),
                    "retrieval": "hybrid",
                }
            )

        if not sources:
            return {
                "success": False,
                "query": query,
                "document": file_name,
                "error": (
                    "No relevant content "
                    "was found in the "
                    "uploaded document."
                ),
                "sources": [],
            }

        return {
            "success": True,
            "query": query,
            "document": file_name,
            "search_type": "hybrid",
            "retrieved": len(
                sources
            ),
            "sources": sources,
        }

    except Exception as error:
        return {
            "success": False,
            "query": query,
            "document": file_name,
            "error": str(error),
            "sources": [],
        }