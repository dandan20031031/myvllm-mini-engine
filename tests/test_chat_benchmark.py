"""Measurement boundaries must not fabricate speedups from incompatible output."""

from benchmarks.benchmark_chat_graph import percentiles, summarize_case


def row(tokens, latency):
    return {"output_tokens": tokens, "total_latency_ms": latency, "outer_call_wall_ms": latency + 1,
            "ttft_ms": 1.0, "tpot_ms": latency / max(tokens - 1, 1)}


def test_summary_uses_total_actual_tokens_over_total_latency():
    pairs = [{"eager": row(2, 10), "cuda_graph": row(2, 5), "outputs_match": True},
             {"eager": row(8, 90), "cuda_graph": row(8, 45), "outputs_match": True}]
    summary = summarize_case(pairs)
    assert summary["modes"]["eager"]["total_tokens_per_second"] == 100
    assert summary["graph_over_eager_throughput_ratio"] == 2
    pairs[1]["outputs_match"] = False
    assert summarize_case(pairs)["graph_over_eager_throughput_ratio"] is None
    assert summarize_case(pairs, comparison_allowed=False)["comparison_allowed"] is False


def test_one_token_requests_do_not_contribute_a_decode_interval():
    summary = summarize_case([{"eager": row(1, 2), "cuda_graph": row(1, 2), "outputs_match": True}])
    assert summary["modes"]["cuda_graph"]["tpot_ms"] == {"count": 0, "p50": None, "p95": None}
    assert percentiles([1, 2, 3])["p50"] == 2
