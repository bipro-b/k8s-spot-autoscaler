"""Phase 3 Step 7: Honest verdict + significance tests + exports JSON for the chart.

Reads:
  results/phase3_sim_seeds.csv   (25-seed sim study)
  results/phase3_real_seeds.csv  (3 real-cluster seeds, optional)

Writes:
  results/phase3_verdict.json    (chart data for the HTML artifact)
  results/phase3_significance.txt

Run: .venv\Scripts\python.exe experiments\phase3_verdict.py
"""
import io, sys, os, json, numpy as np, pandas as pd
from scipy import stats as scipy_stats

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
SLO_P95 = 0.200
SLO_SF  = 0.10
POLICIES = ["hpa_ondemand", "naive_spot", "rl_homogeneous", "rl_spot"]


def load_summary(path):
    df = pd.read_csv(path)
    # Ensure slo_violation exists
    if "slo_violation" not in df.columns:
        p95 = df.get("p95_client", df.get("p95", pd.Series(dtype=float)))
        sf  = df.get("shortfall_frac", 0)
        df["slo_violation"] = ((p95 > SLO_P95) | (sf > SLO_SF)).astype(int)
    return df


def episode_metrics(grp):
    p95_col = "p95_client" if "p95_client" in grp.columns else "p95"
    sf_col  = "shortfall_frac"
    viol    = float(grp["slo_violation"].mean())
    cost    = float(grp["cost_per_hr"].mean()) if "cost_per_hr" in grp.columns else float("nan")
    p95_max = float(grp[p95_col].max()) if p95_col in grp.columns else float("nan")
    pods    = float(grp["pods"].mean()) if "pods" in grp.columns else float("nan")
    sf_mean = float(grp[sf_col].mean()) if sf_col in grp.columns else 0.0
    spot_f  = float(grp["pods_spot"].mean() / max(grp["pods"].mean(), 1e-9)) \
              if "pods_spot" in grp.columns else 0.0

    # Post-eviction metric (window of 3 steps after each eviction)
    post_p95 = float("nan"); post_sf = float("nan"); evict_n = 0
    if "evicted" in grp.columns:
        grp = grp.reset_index(drop=True)
        evict_idx = grp.index[grp["evicted"] > 0].tolist()
        evict_n   = len(evict_idx)
        if evict_idx and p95_col in grp.columns:
            peaks = [float(grp.loc[max(0,i):i+3, p95_col].max()) for i in evict_idx]
            sfs   = [float(grp.loc[max(0,i):i+3, sf_col].max())
                     for i in evict_idx if sf_col in grp.columns]
            post_p95 = float(np.nanmean(peaks))
            post_sf  = float(np.nanmean(sfs)) if sfs else float("nan")

    return dict(viol_rate=viol, cost_hr=cost, p95_max=p95_max, pods_mean=pods,
                sf_rate=sf_mean, spot_frac=spot_f, evict_count=evict_n,
                post_evict_p95=post_p95, post_evict_sf=post_sf)


def paired_test(a_arr, b_arr, name):
    a = np.array(a_arr, dtype=float); b = np.array(b_arr, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    if len(a) < 3:
        print(f"  {name}: n={len(a)} (insufficient — need ≥3)", flush=True)
        return None
    diff = a - b
    t, p = scipy_stats.ttest_rel(a, b)
    se   = np.std(diff, ddof=1) / np.sqrt(len(diff))
    tc   = scipy_stats.t.ppf(0.975, df=len(diff)-1)
    ci   = (float(np.mean(diff)-tc*se), float(np.mean(diff)+tc*se))
    d    = float(np.mean(diff) / (np.std(diff, ddof=1) + 1e-9))
    sig  = "***p<0.01" if p < 0.01 else ("*p<0.05" if p < 0.05 else "n.s.")
    print(f"  {name}:", flush=True)
    print(f"    n={len(a)}  mean_diff={np.mean(diff):+.4f}  "
          f"95%CI=[{ci[0]:+.4f},{ci[1]:+.4f}]  t={t:.2f}  p={p:.4f}  d={d:.2f}  {sig}",
          flush=True)
    return dict(name=name, n=len(a), mean_diff=float(np.mean(diff)),
                ci_lo=ci[0], ci_hi=ci[1], t=float(t), p=float(p), cohens_d=d, sig=sig)


def main():
    sim_path  = os.path.join(RESULTS, "phase3_sim_seeds.csv")
    real_path = os.path.join(RESULTS, "phase3_real_seeds.csv")

    if not os.path.exists(sim_path):
        print(f"ERROR: {sim_path} not found. Run phase3_sim_study.py first.", flush=True)
        sys.exit(1)

    sim_raw = pd.read_csv(sim_path)

    # Detect if CSV is pre-aggregated (one row per seed×policy) or step-level
    if "viol_rate" in sim_raw.columns:
        # Already aggregated by phase3_sim_study.py
        sim_agg = sim_raw.copy()
        if "pods_mean" in sim_agg.columns and "pods" not in sim_agg.columns:
            sim_agg["pods"] = sim_agg["pods_mean"]
    else:
        # Step-level: re-aggregate
        sim_raw = load_summary(sim_path)
        sim_agg = sim_raw.groupby(["policy","seed"]).apply(
            lambda g: pd.Series(episode_metrics(g))).reset_index()

    # ── Sim headline table ────────────────────────────────────────────────────
    print("="*72, flush=True)
    print("PHASE 3 SIM STUDY — 25 seeds — 4-WAY COMPARISON", flush=True)
    print("="*72, flush=True)
    print(f"  {'policy':18s}  {'viol':>7}  {'cost':>8}  {'spot_frac':>10}  "
          f"{'evict':>6}  {'pe_p95':>8}", flush=True)
    print("  " + "-"*65, flush=True)

    sim_stats = {}
    for pol in POLICIES:
        g = sim_agg[sim_agg.policy == pol].copy()
        if g.empty:
            print(f"  {pol:18s}  (no data)", flush=True)
            continue
        avail = [c for c in ["viol_rate","cost_hr","spot_frac","evict_count","post_evict_p95"]
                 if c in g.columns]
        s = {c: float(g[c].mean()) for c in avail}
        for c in ["viol_rate","cost_hr","spot_frac","evict_count","post_evict_p95"]:
            s.setdefault(c, float("nan"))
        s_std = {c: float(g[c].std()) for c in ["viol_rate","cost_hr"] if c in g.columns}
        per_seed_cols = [c for c in ["seed","viol_rate","cost_hr","spot_frac","post_evict_p95"]
                         if c in g.columns]
        sim_stats[pol] = dict(mean=s, std=s_std,
                               per_seed=g[per_seed_cols].replace({float("nan"): None}).to_dict("records"))
        pe = f"{s['post_evict_p95']:.3f}s" if not np.isnan(s["post_evict_p95"]) else "N/A"
        ec = s["evict_count"] if not np.isnan(s["evict_count"]) else 0
        print(f"  {pol:18s}  {s['viol_rate']:>7.1%}  ${s['cost_hr']:>6.2f}/hr  "
              f"{s['spot_frac']:>10.0%}  {ec:>6.1f}  {pe:>8}", flush=True)

    # ── Sim paired tests ──────────────────────────────────────────────────────
    print("\n" + "="*72, flush=True)
    print("SIM PAIRED TESTS", flush=True)
    print("="*72, flush=True)

    def get(pol, col):
        return sim_agg[sim_agg.policy==pol][col].values

    sig_results = []

    print("\n  COST: rl_spot vs hpa_ondemand (positive = hpa more expensive)", flush=True)
    r = paired_test(get("hpa_ondemand","cost_hr"), get("rl_spot","cost_hr"),
                    "hpa_cost − rl_spot_cost")
    if r: sig_results.append(r)

    print("\n  COST: rl_spot vs naive_spot (negative = rl_spot more expensive)", flush=True)
    r = paired_test(get("naive_spot","cost_hr"), get("rl_spot","cost_hr"),
                    "naive_cost − rl_spot_cost")
    if r: sig_results.append(r)

    print("\n  POST-EVICTION P95: naive_spot vs rl_spot (positive = naive worse)", flush=True)
    ns_pe = get("naive_spot","post_evict_p95")
    rs_pe = get("rl_spot",   "post_evict_p95")
    r = paired_test(ns_pe, rs_pe, "naive_pe_p95 − rl_spot_pe_p95")
    if r: sig_results.append(r)

    print("\n  OVERALL SLO: hpa_ondemand vs rl_spot (positive = hpa more violations)", flush=True)
    r = paired_test(get("hpa_ondemand","viol_rate"), get("rl_spot","viol_rate"),
                    "hpa_viol − rl_spot_viol")
    if r: sig_results.append(r)

    print("\n  OVERALL SLO: naive_spot vs rl_spot (positive = naive more violations)", flush=True)
    r = paired_test(get("naive_spot","viol_rate"), get("rl_spot","viol_rate"),
                    "naive_viol − rl_spot_viol")
    if r: sig_results.append(r)

    # ── Real cluster (optional) ───────────────────────────────────────────────
    real_stats = {}
    if os.path.exists(real_path):
        real_raw = load_summary(real_path)
        real_agg = real_raw.groupby(["policy","seed"]).apply(
            lambda g: pd.Series(episode_metrics(g))).reset_index()
        print("\n" + "="*72, flush=True)
        print("REAL CLUSTER (3 seeds)", flush=True)
        print("="*72, flush=True)
        for pol in POLICIES:
            g = real_agg[real_agg.policy == pol]
            if g.empty: continue
            s = {c: float(g[c].mean()) for c in
                 ["viol_rate","cost_hr","spot_frac","evict_count","post_evict_p95"]}
            real_stats[pol] = s
            pe = f"{s['post_evict_p95']:.3f}s" if not np.isnan(s["post_evict_p95"]) else "N/A"
            print(f"  {pol:18s}  viol={s['viol_rate']:.1%}  "
                  f"cost=${s['cost_hr']:.2f}/hr  spot={s['spot_frac']:.0%}  pe_p95={pe}",
                  flush=True)

    # ── Honest verdict ────────────────────────────────────────────────────────
    print("\n" + "="*72, flush=True)
    print("HONEST VERDICT — Phase 3", flush=True)
    print("="*72, flush=True)

    rl_s   = sim_stats.get("rl_spot",   {}).get("mean",{})
    hpa_od = sim_stats.get("hpa_ondemand",{}).get("mean",{})
    naive  = sim_stats.get("naive_spot", {}).get("mean",{})

    cost_save_vs_hpa   = hpa_od.get("cost_hr",0) - rl_s.get("cost_hr",0)
    cost_save_vs_naive = naive.get("cost_hr",0)  - rl_s.get("cost_hr",0)
    pe_improve         = naive.get("post_evict_p95",0) - rl_s.get("post_evict_p95",0)
    slo_vs_hpa         = hpa_od.get("viol_rate",0) - rl_s.get("viol_rate",0)
    slo_vs_naive       = naive.get("viol_rate",0)  - rl_s.get("viol_rate",0)

    if not np.isnan(cost_save_vs_hpa):
        print(f"\n  1. COST vs HPA-ondemand: rl_spot {'saves' if cost_save_vs_hpa>0 else 'costs'} "
              f"${abs(cost_save_vs_hpa):.2f}/hr ({abs(cost_save_vs_hpa)/max(hpa_od.get('cost_hr',1),1e-9):.1%})", flush=True)
    if not np.isnan(cost_save_vs_naive):
        print(f"  2. COST vs naive_spot:   rl_spot {'saves' if cost_save_vs_naive>0 else 'costs more by'} "
              f"${abs(cost_save_vs_naive):.2f}/hr", flush=True)
    if not np.isnan(pe_improve):
        print(f"  3. POST-EVICTION p95 improvement vs naive_spot: {pe_improve:+.3f}s "
              f"({'rl_spot BETTER' if pe_improve>0 else 'rl_spot WORSE'})", flush=True)
    if not np.isnan(slo_vs_hpa):
        print(f"  4. SLO vs HPA-ondemand: {slo_vs_hpa:+.1%} "
              f"({'rl_spot more violations' if slo_vs_hpa<0 else 'rl_spot fewer violations'})", flush=True)
    if not np.isnan(slo_vs_naive):
        print(f"  5. SLO vs naive_spot: {slo_vs_naive:+.1%} "
              f"({'rl_spot better' if slo_vs_naive>0 else 'rl_spot worse'})", flush=True)

    # Claim check
    print("\n  THESIS CLAIM CHECK (25-seed sim):", flush=True)
    cost_claim = cost_save_vs_hpa > 0.5 if not np.isnan(cost_save_vs_hpa) else False
    reclaim_claim = pe_improve > 0.01 if not np.isnan(pe_improve) else False
    cost_sig  = any(r and "cost" in r.get("name","") and r.get("p",1)<0.05 for r in sig_results)
    recl_sig  = any(r and "pe_p95" in r.get("name","") and r.get("p",1)<0.05 for r in sig_results)
    print(f"    (a) rl_spot significantly cheaper than HPA-ondemand? "
          f"{'YES' if (cost_claim and cost_sig) else 'PARTIAL' if cost_claim else 'NO'}", flush=True)
    print(f"    (b) rl_spot significantly better eviction handling than naive_spot? "
          f"{'YES' if (reclaim_claim and recl_sig) else 'PARTIAL' if reclaim_claim else 'NO'}", flush=True)

    # ── Export JSON for chart artifact ────────────────────────────────────────
    chart_data = {
        "sim_per_seed": sim_agg.replace({float("nan"): None}).to_dict("records"),
        "sim_stats":    sim_stats,
        "real_stats":   real_stats,
        "sig_tests":    [r for r in sig_results if r],
        "policies": {
            "hpa_ondemand":  {"label": "HPA on-demand", "color": "#2a78d6"},
            "naive_spot":    {"label": "Naive-spot",    "color": "#eda100"},
            "rl_homogeneous":{"label": "RL homogeneous","color": "#1baf7a"},
            "rl_spot":       {"label": "RL-spot",       "color": "#008300"},
        }
    }
    def sanitize(obj):
        if isinstance(obj, float):
            return None if (obj != obj or obj == float("inf") or obj == float("-inf")) else obj
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitize(v) for v in obj]
        return obj

    verdict_path = os.path.join(RESULTS, "phase3_verdict.json")
    with open(verdict_path, "w", encoding="utf-8") as f:
        json.dump(sanitize(chart_data), f, indent=2)
    print(f"\nSaved -> results/phase3_verdict.json", flush=True)


if __name__ == "__main__":
    main()
