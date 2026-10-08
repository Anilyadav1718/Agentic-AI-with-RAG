import hashlib
import re

import chromadb

from pypdf import PdfReader

from rank_bm25 import BM25Okapi

from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

CHROMA_PATH = "chroma_db"

COLLECTION_NAME = "anil_ai_documents"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# Final number of chunks given to LLM
FINAL_TOP_K = 3


# Candidate pools before fusion
SEMANTIC_CANDIDATES = 10

KEYWORD_CANDIDATES = 10


# Reciprocal Rank Fusion constant
RRF_K = 60


# ============================================================
# EMBEDDING MODEL
# ============================================================

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)


# ============================================================
# CHROMA CLIENT
# ============================================================

chroma_client = chromadb.PersistentClient(
    path=CHROMA_PATH
)


collection = (
    chroma_client
    .get_or_create_collection(
        name=COLLECTION_NAME,

        metadata={
            "hnsw:space": "cosine"
        }
    )
)


# ============================================================
# ACTIVE DOCUMENT
# ============================================================

_active_document = None


def set_active_document(
    file_name: str
):

    global _active_document

    _active_document = file_name


def get_active_document():

    return _active_document


def clear_active_document():

    global _active_document

    _active_document = None


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize(
    text: str
):

    if not text:
        return []


    return re.findall(
        r"[a-zA-Z0-9_]+",
        text.lower()
    )


# ============================================================
# CHUNK TEXT
# ============================================================

def chunk_text(
    text: str,
    chunk_size: int = 900,
    overlap: int = 150
):

    text = (
        text
        .replace("\x00", " ")
        .strip()
    )


    if not text:
        return []


    chunks = []

    start = 0

    text_length = len(text)


    while start < text_length:

        end = min(
            start + chunk_size,
            text_length
        )


        chunk = (
            text[start:end]
            .strip()
        )


        if chunk:

            chunks.append(
                chunk
            )


        if end >= text_length:
            break


        start = end - overlap


    return chunks


# ============================================================
# DELETE EXISTING DOCUMENT
# ============================================================

def delete_document(
    file_name: str
):

    try:

        collection.delete(
            where={
                "file": file_name
            }
        )

    except Exception:

        pass


# ============================================================
# PROCESS PDF
# ============================================================

def process_pdf(
    pdf_path: str,
    file_name: str
):

    try:

        reader = PdfReader(
            pdf_path
        )


        total_pages = len(
            reader.pages
        )


        if total_pages == 0:

            return {
                "success": False,
                "error": (
                    "The PDF contains no pages."
                )
            }


        # Delete old index for same PDF
        delete_document(
            file_name
        )


        documents = []

        metadatas = []

        ids = []


        # ====================================================
        # READ PAGES
        # ====================================================

        for page_index, page in enumerate(
            reader.pages
        ):

            try:

                page_text = (
                    page.extract_text()
                    or ""
                )

            except Exception:

                page_text = ""


            if not page_text.strip():

                continue


            page_number = (
                page_index + 1
            )


            page_chunks = chunk_text(
                page_text
            )


            # =================================================
            # CREATE CHUNKS
            # =================================================

            for chunk_index, chunk in enumerate(
                page_chunks
            ):

                raw_id = (
                    f"{file_name}-"
                    f"{page_number}-"
                    f"{chunk_index}-"
                    f"{chunk}"
                )


                chunk_id = (
                    hashlib
                    .sha256(
                        raw_id.encode(
                            "utf-8"
                        )
                    )
                    .hexdigest()
                )


                documents.append(
                    chunk
                )


                metadatas.append(
                    {
                        "file":
                            file_name,

                        "page":
                            page_number,

                        "chunk":
                            chunk_index,

                        "chunk_id":
                            chunk_id
                    }
                )


                ids.append(
                    chunk_id
                )


        # ====================================================
        # CHECK CONTENT
        # ====================================================

        if not documents:

            return {
                "success": False,
                "error": (
                    "No readable text was "
                    "found in the PDF."
                )
            }


        print(
            "\n===================================="
        )

        print(
            "🧠 GENERATING EMBEDDINGS"
        )

        print(
            "===================================="
        )


        # ====================================================
        # GENERATE EMBEDDINGS
        # ====================================================

        embeddings = (
            embedding_model
            .encode(
                documents,
                show_progress_bar=False,
                normalize_embeddings=True
            )
            .tolist()
        )


        # ====================================================
        # STORE IN CHROMA
        # ====================================================

        batch_size = 100


        for start in range(
            0,
            len(documents),
            batch_size
        ):

            end = start + batch_size


            collection.add(

                ids=
                    ids[start:end],

                documents=
                    documents[start:end],

                metadatas=
                    metadatas[start:end],

                embeddings=
                    embeddings[start:end]
            )


        set_active_document(
            file_name
        )


        print(
            "\n===================================="
        )

        print(
            "📚 PDF INDEXED"
        )

        print(
            "===================================="
        )

        print(
            f"File: {file_name}"
        )

        print(
            f"Pages: {total_pages}"
        )

        print(
            f"Chunks: {len(documents)}"
        )


        return {
            "success": True,
            "file": file_name,
            "pages": total_pages,
            "chunks": len(documents)
        }


    except Exception as error:

        print(
            f"❌ PDF ERROR: {error}"
        )


        return {
            "success": False,
            "error": str(error)
        }


# ============================================================
# GET ALL DOCUMENT CHUNKS
# ============================================================

def get_document_chunks(
    file_name: str
):

    try:

        results = collection.get(

            where={
                "file": file_name
            },

            include=[
                "documents",
                "metadatas"
            ]
        )


        documents = (
            results.get(
                "documents",
                []
            )
            or []
        )


        metadatas = (
            results.get(
                "metadatas",
                []
            )
            or []
        )


        ids = (
            results.get(
                "ids",
                []
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

                else metadata.get(
                    "chunk_id"
                )
            )


            chunks.append(
                {
                    "id":
                        chunk_id,

                    "text":
                        document,

                    "page":
                        metadata.get(
                            "page",
                            "?"
                        ),

                    "file":
                        metadata.get(
                            "file",
                            file_name
                        )
                }
            )


        return chunks


    except Exception as error:

        print(
            f"❌ Chunk loading error: "
            f"{error}"
        )

        return []


# ============================================================
# SEMANTIC SEARCH
# ============================================================

def semantic_search(
    query: str,
    file_name: str,
    top_k: int = SEMANTIC_CANDIDATES
):

    try:

        # ====================================================
        # QUERY EMBEDDING
        # ====================================================

        query_embedding = (
            embedding_model
            .encode(
                [query],
                show_progress_bar=False,
                normalize_embeddings=True
            )
            .tolist()
        )


        # ====================================================
        # VECTOR SEARCH
        # ====================================================

        results = collection.query(

            query_embeddings=
                query_embedding,

            n_results=
                top_k,

            where={
                "file": file_name
            },

            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )


        if (
            not results.get(
                "documents"
            )
            or
            not results[
                "documents"
            ][0]
        ):

            return []


        documents = (
            results[
                "documents"
            ][0]
        )


        metadatas = (
            results[
                "metadatas"
            ][0]
        )


        distances = (
            results[
                "distances"
            ][0]
        )


        ids = (
            results[
                "ids"
            ][0]
        )


        semantic_results = []


        # ====================================================
        # BUILD RESULTS
        # ====================================================

        for index, document in enumerate(
            documents
        ):

            metadata = (
                metadatas[index]
            )


            distance = float(
                distances[index]
            )


            # Cosine similarity
            semantic_score = (
                1.0 - distance
            )


            semantic_score = max(
                0.0,
                min(
                    semantic_score,
                    1.0
                )
            )


            semantic_results.append(
                {
                    "id":
                        ids[index],

                    "text":
                        document,

                    "page":
                        metadata.get(
                            "page",
                            "?"
                        ),

                    "file":
                        metadata.get(
                            "file",
                            file_name
                        ),

                    "semantic_score":
                        semantic_score,

                    "semantic_rank":
                        index + 1
                }
            )


        return semantic_results


    except Exception as error:

        print(
            f"❌ Semantic search error: "
            f"{error}"
        )

        return []


# ============================================================
# KEYWORD / BM25 SEARCH
# ============================================================

def keyword_search(
    query: str,
    file_name: str,
    top_k: int = KEYWORD_CANDIDATES
):

    try:

        chunks = get_document_chunks(
            file_name
        )


        if not chunks:

            return []


        # ====================================================
        # TOKENIZE DOCUMENT
        # ====================================================

        tokenized_corpus = [

            tokenize(
                chunk["text"]
            )

            for chunk in chunks
        ]


        # ====================================================
        # BM25
        # ====================================================

        bm25 = BM25Okapi(
            tokenized_corpus
        )


        query_tokens = tokenize(
            query
        )


        if not query_tokens:

            return []


        scores = bm25.get_scores(
            query_tokens
        )


        keyword_results = []


        # ====================================================
        # SCORE EACH CHUNK
        # ====================================================

        for index, score in enumerate(
            scores
        ):

            if float(score) <= 0:
                continue


            keyword_results.append(
                {
                    **chunks[index],

                    "keyword_score":
                        float(score)
                }
            )


        # ====================================================
        # SORT
        # ====================================================

        keyword_results.sort(

            key=lambda item:
                item[
                    "keyword_score"
                ],

            reverse=True
        )


        keyword_results = (
            keyword_results[
                :top_k
            ]
        )


        # ====================================================
        # RANK
        # ====================================================

        for index, result in enumerate(
            keyword_results,
            start=1
        ):

            result[
                "keyword_rank"
            ] = index


        return keyword_results


    except Exception as error:

        print(
            f"❌ BM25 search error: "
            f"{error}"
        )

        return []


# ============================================================
# RECIPROCAL RANK FUSION
# ============================================================

def reciprocal_rank_fusion(
    semantic_results,
    keyword_results,
    top_k: int = FINAL_TOP_K
):

    fused = {}


    # ========================================================
    # SEMANTIC RESULTS
    # ========================================================

    for rank, result in enumerate(
        semantic_results,
        start=1
    ):

        chunk_id = result[
            "id"
        ]


        if chunk_id not in fused:

            fused[
                chunk_id
            ] = {

                "id":
                    chunk_id,

                "text":
                    result["text"],

                "page":
                    result["page"],

                "file":
                    result["file"],

                "semantic_score":
                    result.get(
                        "semantic_score",
                        0
                    ),

                "keyword_score":
                    0.0,

                "semantic_rank":
                    rank,

                "keyword_rank":
                    None,

                "rrf_score":
                    0.0
            }


        fused[
            chunk_id
        ][
            "rrf_score"
        ] += (

            1.0
            /
            (
                RRF_K
                + rank
            )
        )


    # ========================================================
    # KEYWORD RESULTS
    # ========================================================

    for rank, result in enumerate(
        keyword_results,
        start=1
    ):

        chunk_id = result[
            "id"
        ]


        if chunk_id not in fused:

            fused[
                chunk_id
            ] = {

                "id":
                    chunk_id,

                "text":
                    result["text"],

                "page":
                    result["page"],

                "file":
                    result["file"],

                "semantic_score":
                    0.0,

                "keyword_score":
                    result.get(
                        "keyword_score",
                        0
                    ),

                "semantic_rank":
                    None,

                "keyword_rank":
                    rank,

                "rrf_score":
                    0.0
            }


        else:

            fused[
                chunk_id
            ][
                "keyword_score"
            ] = result.get(
                "keyword_score",
                0
            )


            fused[
                chunk_id
            ][
                "keyword_rank"
            ] = rank


        fused[
            chunk_id
        ][
            "rrf_score"
        ] += (

            1.0
            /
            (
                RRF_K
                + rank
            )
        )


    # ========================================================
    # SORT FUSED RESULTS
    # ========================================================

    final_results = list(
        fused.values()
    )


    final_results.sort(

        key=lambda item:
            item[
                "rrf_score"
            ],

        reverse=True
    )


    final_results = (
        final_results[
            :top_k
        ]
    )


    # ========================================================
    # DISPLAY RELEVANCE
    # ========================================================

    if final_results:

        max_rrf = max(

            item[
                "rrf_score"
            ]

            for item in final_results
        )

    else:

        max_rrf = 0


    for result in final_results:

        # ====================================================
        # HYBRID RELEVANCE
        #
        # Relative RRF score for UI.
        # NOT probability/confidence.
        # ====================================================

        if max_rrf > 0:

            hybrid_score = (

                result[
                    "rrf_score"
                ]

                / max_rrf
            )

        else:

            hybrid_score = 0


        result[
            "hybrid_relevance"
        ] = round(
            hybrid_score * 100,
            1
        )


        # ====================================================
        # SEMANTIC RELEVANCE
        # ====================================================

        result[
            "semantic_relevance"
        ] = round(

            result.get(
                "semantic_score",
                0
            )
            * 100,

            1
        )


        # ====================================================
        # MATCH TYPE
        # ====================================================

        has_semantic = (
            result.get(
                "semantic_rank"
            )
            is not None
        )


        has_keyword = (
            result.get(
                "keyword_rank"
            )
            is not None
        )


        if (
            has_semantic
            and
            has_keyword
        ):

            match_type = (
                "Semantic + Keyword"
            )

        elif has_semantic:

            match_type = (
                "Semantic"
            )

        else:

            match_type = (
                "Keyword"
            )


        result[
            "match_type"
        ] = match_type


    return final_results


# ============================================================
# HYBRID SEARCH
# ============================================================

def hybrid_search(
    query: str,
    file_name: str,
    top_k: int = FINAL_TOP_K
):

    print(
        "\n===================================="
    )

    print(
        "🔎 HYBRID SEARCH"
    )

    print(
        "===================================="
    )

    print(
        f"Query: {query}"
    )


    # ========================================================
    # SEMANTIC
    # ========================================================

    semantic_results = (
        semantic_search(

            query=query,

            file_name=file_name,

            top_k=SEMANTIC_CANDIDATES
        )
    )


    print(
        f"🧠 Semantic candidates: "
        f"{len(semantic_results)}"
    )


    # ========================================================
    # KEYWORD
    # ========================================================

    keyword_results = (
        keyword_search(

            query=query,

            file_name=file_name,

            top_k=KEYWORD_CANDIDATES
        )
    )


    print(
        f"🔤 BM25 candidates: "
        f"{len(keyword_results)}"
    )


    # ========================================================
    # FUSION
    # ========================================================

    final_results = (
        reciprocal_rank_fusion(

            semantic_results=
                semantic_results,

            keyword_results=
                keyword_results,

            top_k=
                top_k
        )
    )


    print(
        "\n🏆 TOP HYBRID RESULTS"
    )


    for index, result in enumerate(
        final_results,
        start=1
    ):

        print(
            f"{index}. "
            f"Page {result['page']} | "
            f"Hybrid {result['hybrid_relevance']}% | "
            f"Semantic {result['semantic_relevance']}% | "
            f"BM25 {result['keyword_score']:.3f} | "
            f"{result['match_type']}"
        )


    return final_results


# ============================================================
# KNOWLEDGE BASE TOOL
# ============================================================

def knowledge_base_search(
    query: str
):

    file_name = (
        get_active_document()
    )


    if not file_name:

        return {
            "success": False,
            "error": (
                "No uploaded PDF knowledge "
                "base is currently active."
            )
        }


    # ========================================================
    # HYBRID RETRIEVAL
    # ========================================================

    results = hybrid_search(

        query=query,

        file_name=file_name,

        top_k=FINAL_TOP_K
    )


    if not results:

        return {
            "success": False,

            "document":
                file_name,

            "error":
                (
                    "No relevant information "
                    "was found in the document."
                )
        }


    # ========================================================
    # SOURCES
    # ========================================================

    sources = []


    for result in results:

        sources.append(
            {
                "file":
                    result["file"],

                "page":
                    result["page"],

                "text":
                    result["text"],

                "relevance":
                    result[
                        "hybrid_relevance"
                    ],

                "semantic_relevance":
                    result[
                        "semantic_relevance"
                    ],

                "keyword_score":
                    round(
                        result[
                            "keyword_score"
                        ],
                        3
                    ),

                "match_type":
                    result[
                        "match_type"
                    ],

                "retrieval":
                    "hybrid"
            }
        )


    return {
        "success": True,

        "query":
            query,

        "document":
            file_name,

        "search_type":
            "hybrid",

        "retrieved":
            len(sources),

        "sources":
            sources
    }