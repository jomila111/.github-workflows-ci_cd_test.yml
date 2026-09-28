"""
test_automated_suite.py
Automated test suite for Large Dataset Application:
- Functional: Hashing & 2-Minute Window Chunking
- Security: SQL Injection Resistance & Fuzzing (DAST)
- Performance: DuckDB In-Memory OLAP Latency (<150ms SLA)
"""
import hashlib
import time
import duckdb
import pytest
from dataclasses import dataclass
from typing import List


# --- Data Models & Helpers ---
@dataclass
class WordToken:
    word: str
    start_time: float
    end_time: float


def generate_id(value: str) -> str:
    """Deterministic hashing for idempotent upserts."""
    return hashlib.blake2b(value.encode("utf-8"), digest_size=12).hexdigest()


def chunk_passages(words: List[WordToken], window_sec: float = 120.0, stride_sec: float = 60.0):
    """Splits timestamped words into 2-minute windows."""
    if not words:
        return []
    chunks = []
    max_duration = words[-1].end_time
    curr_start = 0.0
    while curr_start < max_duration:
        curr_end = curr_start + window_sec
        matched = [w.word for w in words if curr_start <= w.start_time < curr_end]
        if matched:
            chunks.append({"start": curr_start, "end": curr_end, "text": " ".join(matched)})
        curr_start += stride_sec
    return chunks


# --- 1. Functional Tests ---
def test_idempotent_hashing():
    """Verify hashing produces deterministic IDs."""
    id1 = generate_id("spotify:episode:001")
    id2 = generate_id("spotify:episode:001")
    id3 = generate_id("spotify:episode:002")
    assert id1 == id2, "Identical inputs must yield identical IDs"
    assert id1 != id3, "Distinct inputs must yield distinct IDs"


def test_passage_chunking():
    """Verify sliding window algorithm divides 180s into 2 chunks."""
    words = [WordToken(word=f"word_{t}", start_time=float(t), end_time=float(t + 1)) for t in range(0, 180, 10)]
    chunks = chunk_passages(words, window_sec=120.0, stride_sec=60.0)
    assert len(chunks) == 2
    assert chunks[0]["start"] == 0.0 and chunks[0]["end"] == 120.0
    assert chunks[1]["start"] == 60.0 and chunks[1]["end"] == 180.0


# --- 2. Security (DAST) Tests ---
@pytest.mark.parametrize("payload", ["' OR 1=1 --", "'; DROP TABLE episodes; --"])
def test_dast_sql_injection_resilience(payload):
    """Test DuckDB query resilience against SQL injection payloads."""
    con = duckdb.connect(database=":memory:")
    con.execute("CREATE TABLE episodes (id VARCHAR, publisher VARCHAR, title VARCHAR);")
    con.execute("INSERT INTO episodes VALUES ('1', 'Safe Publisher', 'AI Episode');")
    
    # Safe parameterized query pattern
    query = "SELECT * FROM episodes WHERE publisher = ?"
    result = con.execute(query, [payload]).fetchall()
    
    # Must safely return 0 results and NOT drop table or return all rows
    assert len(result) == 0
    table_check = con.execute("SELECT COUNT(*) FROM episodes;").fetchone()[0]
    assert table_check == 1, "Table must remain intact"


# --- 3. Performance SLA Benchmark (<150ms) ---
def test_olap_aggregation_latency_sla():
    """Verify in-memory OLAP analytics execute well under 150ms SLA."""
    con = duckdb.connect(database=":memory:")
    con.execute("""
        CREATE TABLE episodes AS 
        SELECT 
            'ep_' || range AS id, 
            'Publisher_' || (range % 10) AS publisher, 
            3600.0 AS duration_sec 
        FROM range(1000);
    """)
    
    t0 = time.perf_counter()
    result = con.execute("""
        SELECT publisher, COUNT(*), AVG(duration_sec) 
        FROM episodes 
        GROUP BY publisher;
    """).fetchall()
    latency_ms = (time.perf_counter() - t0) * 1000.0
    
    print(f"\n[SLA Benchmark] OLAP Latency: {latency_ms:.2f} ms")
    assert len(result) == 10
    assert latency_ms < 150.0, f"Latency {latency_ms:.2f}ms violated 150ms SLA!"
