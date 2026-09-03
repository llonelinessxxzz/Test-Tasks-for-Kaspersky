from dataclasses import replace

import numpy as np
import pytest

from support_rag.core.schemas import DocumentChunk
from support_rag.retrieval.index import (
    IndexManifest,
    chunk_ids_fingerprint,
    load_dense_index,
    save_dense_index,
)


@pytest.fixture
def saved_index(tmp_path):
    chunks = [
        DocumentChunk(
            chunk_id="a:0",
            document_id="a",
            title="Article",
            source_url="https://example.com/a",
            text="Text",
            chunk_index=0,
            token_count=1,
        ),
    ]
    matrix = np.array([[0.6, 0.8]], dtype=np.float32)
    manifest = IndexManifest(
        1, "test-model", 2, 512, "search_document:", 1, 1, chunk_ids_fingerprint(chunks)
    )
    save_dense_index(matrix, manifest, tmp_path)
    return chunks, matrix, manifest, tmp_path


def load(chunks, path):
    return load_dense_index(
        chunks,
        path,
        expected_model="test-model",
        expected_max_length=512,
        expected_document_prefix="search_document:",
    )


def test_saved_index_round_trip(saved_index):
    chunks, matrix, _, path = saved_index
    np.testing.assert_array_equal(load(chunks, path), matrix)


@pytest.mark.parametrize(
    "field,value",
    [
        ("embedding_model", "other-model"),
        ("embedding_max_length", 256),
        ("document_prefix", "wrong:"),
        ("chunk_ids_sha256", "stale"),
    ],
)
def test_incompatible_index_is_rejected(saved_index, field, value):
    chunks, matrix, manifest, path = saved_index
    save_dense_index(matrix, replace(manifest, **{field: value}), path)
    with pytest.raises(RuntimeError):
        load(chunks, path)
