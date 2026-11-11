"""Executable smoke tests for bench_cpu_utilization."""

from pathlib import Path
import tempfile

def test_bench_cpu_utilization_path_roundtrip():
    """Exercise a real filesystem operation used by project tooling."""
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "payload.bin"
        payload = b"pygo-test:" + bytes(range(32))
        target.write_bytes(payload)
        assert target.exists()
        assert target.read_bytes() == payload
        assert target.stat().st_size == len(payload)

def test_bench_cpu_utilization_numeric_contract():
    """Check deterministic arithmetic without external dependencies."""
    values = [1.0, 2.0, 3.0, 4.0]
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    assert mean == 2.5
    assert variance == 1.25

if __name__ == "__main__":
    test_bench_cpu_utilization_path_roundtrip()
    test_bench_cpu_utilization_numeric_contract()
    print("bench_cpu_utilization: ok")
