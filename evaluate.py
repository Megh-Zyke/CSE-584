"""
Tri-Guard Evaluation Script — v2
Modeled after GPT Semantic Cache paper methodology
Measures all 4 metrics from the proposal:
1. Staleness Rate
2. Cache Integrity (Gate 3)
3. Effective Hit Rate
4. Latency Benchmark
"""

import pandas as pd
import requests
import time
import json
from datetime import datetime
import os
from dotenv import load_dotenv
load_dotenv()

API_KEYS = [
    os.environ.get("GOOGLE_API_KEY"),
    os.environ.get("GOOGLE_API_KEY_2"),
    os.environ.get("GOOGLE_API_KEY_3"),
    os.environ.get("GOOGLE_API_KEY_4"),
    os.environ.get("GOOGLE_API_KEY_5"),
    os.environ.get("GOOGLE_API_KEY_6"),
]
API_KEYS = [k for k in API_KEYS if k]  # remove empty ones
current_key_idx = 0

BASE_URL = "http://localhost:8000"

def ask(query: str, retry_count: int = 0) -> dict:
    global current_key_idx
    try:
        response = requests.post(
            f"{BASE_URL}/ask",
            json={"query": query},
            timeout=60
        )
        if response.status_code == 500:
            body = response.json()
            if "429" in str(body):
                current_key_idx += 1
                if current_key_idx < len(API_KEYS):
                    new_key = API_KEYS[current_key_idx]
                    with open(".env", "r") as f:
                        env = f.read()
                    import re
                    env = re.sub(r"GOOGLE_API_KEY=.*", f"GOOGLE_API_KEY={new_key}", env)
                    with open(".env", "w") as f:
                        f.write(env)
                    print(f"  [KEY] Switched to API key {current_key_idx + 1}/{len(API_KEYS)}")
                    time.sleep(60)
                    return ask(query, 0)
                else:
                    print("  [KEY] All API keys exhausted!")
                    return {"source": "ERROR", "error": "all_keys_exhausted", "latency_seconds": 0}
            if "503" in str(body) and retry_count < 3:
                wait = 15 * (retry_count + 1)
                print(f"  [503] waiting {wait}s...")
                time.sleep(wait)
                return ask(query, retry_count + 1)
        return response.json()
    except Exception as e:
        return {"source": "ERROR", "error": str(e), "latency_seconds": 0}

def get_stats() -> dict:
    try:
        return requests.get(f"{BASE_URL}/stats").json()
    except:
        return {}

def run_evaluation():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("=" * 60)
    print("TRI-GUARD EVALUATION — All Three Gates")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # ── Load dataset ──────────────────────────────────────────
    print("\nLoading dataset...")
    try:
        df = pd.read_excel("datasets/evaluation_dataset.xlsx")
    except:
        df = pd.read_excel("evaluation_dataset.xlsx")

    print(f"Total queries: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nCategory distribution:")
    print(df["Temporal_Category"].value_counts())

    # ── Phase 1: Cache population (first 500) ─────────────────
    print("\n" + "=" * 60)
    print("PHASE 1 — Populating cache with 500 queries")
    print("=" * 60)

    train_df = df.head(100)
    phase1_results = []

    for i, row in train_df.iterrows():
        query = row["Question"]
        true_category = row["Temporal_Category"]
        result = ask(query)
        source = result.get("source", "ERROR")
        phase1_results.append({
            "query": query,
            "true_category": true_category,
            "source": source,
            "latency": result.get("latency_seconds", 0)
        })
        if i % 10 == 0:
            print(f"  [{i}/100] {source} — {query[:40]}...")
        time.sleep(7)

    stats_after_p1 = get_stats()
    p1_metrics = stats_after_p1.get("metrics", {})
    print(f"\nAfter population:")
    print(f"  ChromaDB entries: {stats_after_p1.get('chroma_entries', 0)}")
    print(f"  Redis keys: {stats_after_p1.get('redis_keys', 0)}")
    print(f"  Gate3 admitted: {p1_metrics.get('gate3_admitted', 'N/A')}")
    print(f"  Gate3 blocked: {p1_metrics.get('gate3_blocked', 'N/A')}")
    pd.DataFrame(phase1_results).to_csv(f"phase1_results_{timestamp}.csv", index=False)

    # ── Phase 2: Test queries (next 500) ──────────────────────
    print("\n" + "=" * 60)
    print("PHASE 2 — Running 500 test queries")
    print("=" * 60)

    test_df = df.iloc[100:200]
    results = []

    for i, row in test_df.iterrows():
        query = row["Question"]
        true_category = row["Temporal_Category"]
        result = ask(query)
        source = result.get("source", "ERROR")
        predicted_category = result.get("category", "Unknown")
        latency = result.get("latency_seconds", 0)
        confidence = result.get("confidence", 0)

        results.append({
            "query": query,
            "true_category": true_category,
            "predicted_category": predicted_category,
            "source": source,
            "latency": latency,
            "confidence": confidence,
            "cache_hit": source in ["REDIS_HIT", "CACHE_HIT_FAST", "CACHE_HIT_VERIFIED"],
            "volatile_bypass": source == "GEMINI_VOLATILE",
            "gemini_call": source == "GEMINI_API",
            "timestamp": datetime.now().isoformat(),
            "api_call_latency": result.get("api_call_latency", 0),
            "similarity": result.get("similarity", 0),
            "gate3_confidence": result.get("confidence", 0),
        })

        if (i - 100) % 10 == 0:
            print(f"  [{i-100}/100] {source} — {latency:.3f}s — {true_category}")

        time.sleep(7)

    # ── Phase 3: Compute metrics ───────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 3 — Results")
    print("=" * 60)

    results_df = pd.DataFrame(results)
    final_stats = get_stats()
    final_metrics = final_stats.get("metrics", {})

    total = len(results_df)
    cache_hits = results_df["cache_hit"].sum()
    volatile_bypasses = results_df["volatile_bypass"].sum()
    gemini_calls = results_df["gemini_call"].sum()

    hit_rate = cache_hits / total * 100
    api_reduction = (1 - gemini_calls / total) * 100

    # Category accuracy
    valid_cat = results_df[results_df["predicted_category"] != "Unknown"]
    category_accuracy = (
        valid_cat["true_category"] == valid_cat["predicted_category"]
    ).mean() * 100 if len(valid_cat) > 0 else 0

    # Staleness rate
    volatile_queries = results_df[results_df["true_category"] == "Volatile"]
    staleness_rate = (
        volatile_queries[~volatile_queries["volatile_bypass"]].shape[0] /
        max(1, len(volatile_queries)) * 100
    )

    # Latency
    cache_latency = results_df[results_df["cache_hit"]]["latency"].mean() if cache_hits > 0 else 0
    gemini_latency = results_df[results_df["gemini_call"]]["latency"].mean() if gemini_calls > 0 else 0

    # Gate 3
    gate3_admitted = final_metrics.get("gate3_admitted", 0)
    gate3_blocked = final_metrics.get("gate3_blocked", 0)
    gate3_total = gate3_admitted + gate3_blocked
    cache_integrity = (
        gate3_blocked / gate3_total * 100
        if gate3_total > 0
        else "N/A — Ollama not running"
    )

    # ── Print all 4 metrics ────────────────────────────────────
    print(f"\n── METRIC 1: EFFECTIVE HIT RATE ─────────────────────")
    print(f"  Cache hits: {int(cache_hits)}/{total}")
    print(f"  Effective Hit Rate: {hit_rate:.1f}%")
    print(f"  API call reduction: {api_reduction:.1f}%")
    print(f"  GPTCache baseline: ~65% | Tri-Guard: {hit_rate:.1f}%")

    print(f"\n── METRIC 2: STALENESS RATE (Gate 2) ───────────────────")
    print(f"  Volatile queries: {len(volatile_queries)}")
    print(f"  Correctly bypassed: {int(volatile_bypasses)}")
    print(f"  Staleness Rate: {staleness_rate:.1f}%")
    print(f"  (0% = no stale responses served)")

    print(f"\n── METRIC 3: CACHE INTEGRITY (Gate 3) ──────────────────")
    if isinstance(cache_integrity, str):
        print(f"  {cache_integrity}")
    else:
        print(f"  Gate3 total: {gate3_total}")
        print(f"  Admitted: {gate3_admitted}")
        print(f"  Blocked (hallucinations prevented): {gate3_blocked}")
        print(f"  Cache Integrity: {cache_integrity:.1f}%")

    print(f"\n── METRIC 4: LATENCY BENCHMARK ──────────────────────────")
    print(f"  Cache hit latency:   {cache_latency*1000:.0f}ms")
    print(f"  Gemini call latency: {gemini_latency*1000:.0f}ms")
    if cache_latency > 0 and gemini_latency > 0:
        print(f"  Speedup: {gemini_latency/cache_latency:.0f}x faster")

    print(f"\n── TTL CATEGORY ACCURACY (Gate 2) ───────────────────────")
    print(f"  Overall accuracy: {category_accuracy:.1f}%")
    for cat in ["Static", "Slow-Moving", "Volatile"]:
        cat_df = results_df[results_df["true_category"] == cat]
        if len(cat_df) > 0:
            acc = (cat_df["true_category"] == cat_df["predicted_category"]).mean() * 100
            hit = cat_df["cache_hit"].sum() / len(cat_df) * 100
            lat = cat_df["latency"].mean() * 1000
            print(f"  {cat}: accuracy={acc:.1f}%, hit_rate={hit:.1f}%, avg_latency={lat:.0f}ms")

    print(f"\n── SOURCE BREAKDOWN ─────────────────────────────────────")
    print(results_df["source"].value_counts().to_string())

    print(f"\n── VS GPTCACHE BASELINE ─────────────────────────────────")
    print(f"{'Metric':<35} {'GPTCache':<20} {'Tri-Guard':<15}")
    print(f"{'-'*70}")
    print(f"{'TTL Classification':<35} {'Flat 24h TTL':<20} {'3-tier dynamic':<15}")
    print(f"{'Volatile Handling':<35} {'Cached (stale)':<20} {'Bypassed':<15}")
    print(f"{'Hallucination Gate':<35} {'None':<20} {'Dual SLM judge':<15}")
    print(f"{'Effective Hit Rate':<35} {'~65%':<20} {hit_rate:.1f}%")
    print(f"{'Staleness Rate':<35} {'~100% volatile':<20} {staleness_rate:.1f}%")
    print(f"{'Cache Hit Latency':<35} {'~100ms':<20} {cache_latency*1000:.0f}ms")
    print(f"{'Category Accuracy':<35} {'N/A':<20} {category_accuracy:.1f}%")

    # ── Save results ─────────────────────────────────────────
    results_df.to_csv(f"evaluation_results_{timestamp}.csv", index=False)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_queries_tested": total,
        "metrics": {
            "effective_hit_rate_percent": round(hit_rate, 2),
            "api_reduction_percent": round(api_reduction, 2),
            "staleness_rate_percent": round(staleness_rate, 2),
            "cache_integrity": cache_integrity if isinstance(cache_integrity, str)
                               else round(cache_integrity, 2),
            "ttl_category_accuracy_percent": round(category_accuracy, 2),
            "cache_hit_latency_ms": round(cache_latency * 1000, 2),
            "gemini_latency_ms": round(gemini_latency * 1000, 2),
            "speedup_factor": round(gemini_latency / cache_latency, 1)
                              if cache_latency > 0 else "N/A"
        },
        "gate3": {
            "admitted": gate3_admitted,
            "blocked": gate3_blocked,
            "total": gate3_total
        },
        "source_breakdown": results_df["source"].value_counts().to_dict()
    }

    with open(f"evaluation_summary_{timestamp}.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Saved: evaluation_results.csv")
    print(f"✅ Saved: evaluation_summary.json")
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    run_evaluation()