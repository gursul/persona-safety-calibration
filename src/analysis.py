"""Phase 4 analysis: persona effects on over-refusal, under-refusal and calibration.

Label scenarios (see PLAN.md):
  primary    deflection is not a refusal; close calls as labeled
  strict     deflection and close-call compliances count as refusal
  lenient    close-call refusals count as compliance
  judge_only the original judge labels (partial merged into refusal), for reference

Outcomes: over-refusal = refusal on SAFE prompts; under-refusal = compliance on UNSAFE
prompts; discrimination J = refusal rate on unsafe minus refusal rate on safe prompts.
Prompts are the unit of resampling (each prompt is answered by all six personas).

Usage: python3 src/analysis.py [--boot 2000] [--firth-boot 1000] [--sims 200]
Writes tables and figures to results/.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.contingency_tables import cochrans_q, mcnemar
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
PERSONAS = ["neutral", "childrens_tutor", "medical_professional",
            "cybersecurity_professional", "creative_fiction_writer", "customer_support_agent"]
SCENARIOS = ["primary", "strict", "lenient", "judge_only"]
SEED = 2026


def load() -> pd.DataFrame:
    d = pd.read_csv(ROOT / "data/judged/full_corrected.csv", keep_default_na=False)
    d["close_call"] = d.close_call.astype(str) == "True"
    return d


def scenario_labels(d: pd.DataFrame, scn: str) -> pd.Series:
    if scn == "judge_only":
        return d.judge2.map({"full_compliance": "C", "full_refusal": "R"})
    y = d.final_label3.map({"full_compliance": "C", "full_refusal": "R", "deflection": "D"})
    if scn == "strict":
        y = y.where(y != "D", "R")
        y = y.mask(d.close_call & (d.final_label3 == "full_compliance"), "R")
    if scn == "lenient":
        y = y.mask(d.close_call & (d.final_label3 == "full_refusal"), "C")
    return y


def matrices(d: pd.DataFrame, y: pd.Series) -> dict:
    """Prompt x persona 0/1 matrices for the two prompt strata."""
    t = d.assign(y=y.values)
    out = {}
    for sl in ["safe", "unsafe"]:
        g = t[t.safety_label == sl]
        ids = sorted(g.xstest_id.unique())
        r = g.assign(v=(g.y == "R").astype(int)).pivot(index="xstest_id", columns="persona_id", values="v").loc[ids, PERSONAS]
        c = g.assign(v=(g.y == "C").astype(int)).pivot(index="xstest_id", columns="persona_id", values="v").loc[ids, PERSONAS]
        out[sl] = {"R": r.values, "C": c.values, "ids": ids}
    return out


def point_stats(m: dict) -> dict:
    over = m["safe"]["R"].mean(0)
    under = m["unsafe"]["C"].mean(0)
    refu = m["unsafe"]["R"].mean(0)
    return {"over": over, "under": under, "refuse_unsafe": refu, "J": refu - over}


def bootstrap(m: dict, B: int, rng) -> dict:
    ns, nu = m["safe"]["R"].shape[0], m["unsafe"]["R"].shape[0]
    si = rng.integers(0, ns, (B, ns)); ui = rng.integers(0, nu, (B, nu))
    over = m["safe"]["R"][si].mean(1)
    under = m["unsafe"]["C"][ui].mean(1)
    refu = m["unsafe"]["R"][ui].mean(1)
    return {"over": over, "under": under, "J": refu - over}


def ci(a, q=(2.5, 97.5)):
    return np.percentile(a, q, axis=0)


def wald_boot(est: np.ndarray, boot: np.ndarray, D: np.ndarray | None = None) -> tuple[float, float]:
    """Wald-type statistic with a bootstrap-calibrated p-value.

    Tests that D @ est = 0 (default: the 5 differences vs neutral). The p-value is the share
    of bootstrap statistics (centered at the estimate) at least as large as the observed one,
    so it does not rely on a chi-square approximation (rare-event cells are far from normal).
    """
    if D is None:
        D = np.hstack([-np.ones((5, 1)), np.eye(5)])
    d = D @ est
    bd = boot @ D.T
    Vinv = np.linalg.pinv(np.cov(bd.T))
    stat = float(d @ Vinv @ d)
    dev = bd - d
    sims = np.einsum("ij,jk,ik->i", dev, Vinv, dev)
    return stat, float((1 + (sims >= stat).sum()) / (len(sims) + 1))


def firth_logit(X, y, iters=60, tol=1e-8):
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ b)); w = p * (1 - p)
        XtWX = X.T @ (X * w[:, None]) + 1e-10 * np.eye(X.shape[1])
        inv = np.linalg.inv(XtWX)
        h = w * np.einsum("ij,jk,ik->i", X, inv, X)
        U = X.T @ (y - p + h * (0.5 - p))
        step = inv @ U
        mx = np.abs(step).max()
        if mx > 5: step *= 5 / mx
        b = b + step
        if np.abs(step).max() < tol: break
    return b


def design(personas_idx, unsafe):
    k = len(PERSONAS)
    P = np.zeros((len(personas_idx), k - 1));
    for j in range(1, k): P[:, j - 1] = personas_idx == j
    u = unsafe.astype(float)[:, None]
    return np.hstack([np.ones((len(u), 1)), P, u, P * u])


def firth_interaction(m: dict, B: int, rng) -> dict:
    ns, nu = m["safe"]["R"].shape[0], m["unsafe"]["R"].shape[0]
    pid = np.tile(np.arange(len(PERSONAS)), ns + nu)

    def fit(si, ui):
        y = np.concatenate([m["safe"]["R"][si].ravel(), m["unsafe"]["R"][ui].ravel()])
        un = np.concatenate([np.zeros(len(si) * 6), np.ones(len(ui) * 6)])
        pidx = np.tile(np.arange(6), len(si) + len(ui))
        return firth_logit(design(pidx, un), y.astype(float))

    b0 = fit(np.arange(ns), np.arange(nu))
    bs = []
    for _ in range(B):
        bs.append(fit(rng.integers(0, ns, ns), rng.integers(0, nu, nu)))
    bs = np.array(bs)
    stat, p = wald_boot(b0[7:], bs[:, 7:], np.eye(5))
    return {"coef": b0, "boot": bs, "wald": stat, "p": p}


def h1_tests(m: dict) -> pd.DataFrame:
    rows = []
    for name, key, stratum in [("over-refusal (safe: refused)", "R", "safe"), ("under-refusal (unsafe: complied)", "C", "unsafe")]:
        x = m[stratum][key]
        q = cochrans_q(x)
        rows.append({"outcome": name, "test": "Cochran Q (6 personas)", "persona": "all", "stat": q.statistic, "p_raw": q.pvalue, "n_prompts": len(x)})
        for j in range(1, 6):
            a, b = x[:, 0], x[:, j]
            n10 = int(((a == 1) & (b == 0)).sum()); n01 = int(((a == 0) & (b == 1)).sum())
            tab = [[int(((a == 1) & (b == 1)).sum()), n10], [n01, int(((a == 0) & (b == 0)).sum())]]
            p = mcnemar(tab, exact=True).pvalue if (n10 + n01) > 0 else 1.0
            rows.append({"outcome": name, "test": "McNemar vs neutral", "persona": PERSONAS[j], "neutral_only": n10, "persona_only": n01, "p_raw": p})
    t = pd.DataFrame(rows)
    mask = t.test == "McNemar vs neutral"
    t.loc[mask, "p_holm"] = multipletests(t.loc[mask, "p_raw"], method="holm")[1]
    return t


def rates_table(est: dict, boots: dict, scn: str) -> pd.DataFrame:
    rows = []
    for j, p in enumerate(PERSONAS):
        row = {"scenario": scn, "persona": p}
        for key, name in [("over", "over_refusal"), ("under", "under_refusal"), ("J", "discrimination")]:
            lo, hi = ci(boots[key][:, j])
            row.update({name: est[key][j], name + "_lo": lo, name + "_hi": hi})
            if j > 0:
                dlo, dhi = ci(boots[key][:, j] - boots[key][:, 0])
                row[name + "_diff_vs_neutral"] = est[key][j] - est[key][0]
                row[name + "_diff_lo"], row[name + "_diff_hi"] = dlo, dhi
        rows.append(row)
    return pd.DataFrame(rows)


def run_scenario(d, scn, B, FB, rng):
    y = scenario_labels(d, scn)
    m = matrices(d, y)
    est = point_stats(m); boots = bootstrap(m, B, rng)
    rt = rates_table(est, boots, scn)
    tests = h1_tests(m).assign(scenario=scn)
    wald = {}
    for key in ["over", "under", "J"]:
        wald[key] = wald_boot(est[key], boots[key])
    fi = firth_interaction(m, FB, rng)
    extra = pd.DataFrame([
        {"scenario": scn, "outcome": "over-refusal differences vs neutral", "test": "Wald (bootstrap-calibrated p)", "stat": wald["over"][0], "p_raw": wald["over"][1]},
        {"scenario": scn, "outcome": "under-refusal differences vs neutral", "test": "Wald (bootstrap-calibrated p)", "stat": wald["under"][0], "p_raw": wald["under"][1]},
        {"scenario": scn, "outcome": "H2: discrimination index differs by persona", "test": "Wald (bootstrap-calibrated p)", "stat": wald["J"][0], "p_raw": wald["J"][1]},
        {"scenario": scn, "outcome": "H2: persona x safety interaction (Firth logit)", "test": "joint Wald (bootstrap-calibrated p)", "stat": fi["wald"], "p_raw": fi["p"]},
    ])
    coef = pd.DataFrame({"term": ["intercept"] + [f"persona:{p}" for p in PERSONAS[1:]] + ["unsafe"] + [f"persona:{p} x unsafe" for p in PERSONAS[1:]],
                         "log_odds": fi["coef"], "lo": np.percentile(fi["boot"], 2.5, axis=0), "hi": np.percentile(fi["boot"], 97.5, axis=0)}).assign(scenario=scn)
    return rt, pd.concat([tests, extra]), coef, m, est


def type_rates(d, y) -> pd.DataFrame:
    t = d.assign(y=y.values)
    rows = []
    for (ty, sl), g in t.groupby(["xstest_type", "safety_label"]):
        for p in PERSONAS:
            h = g[g.persona_id == p]
            rows.append({"xstest_type": ty, "safety_label": sl, "persona": p, "n": len(h), "refused": int((h.y == "R").sum()), "complied": int((h.y == "C").sum()), "deflected": int((h.y == "D").sum())})
    return pd.DataFrame(rows)


def type_tests(d: pd.DataFrame) -> pd.DataFrame:
    """Exploratory: do persona differences concentrate in particular XSTest prompt types?
    Cochran's Q within one type vs within all other types (chosen after inspecting the
    per-type counts, so treat as post hoc)."""
    rows = []
    for scn in ["primary", "lenient", "strict"]:
        t = d.assign(y=scenario_labels(d, scn).values)
        for stratum, key, focus in [("unsafe", "C", "contrast_privacy"), ("safe", "R", "safe_contexts")]:
            g = t[t.safety_label == stratum]
            for name, sub in [(focus, g[g.xstest_type == focus]), ("all other " + stratum + " types", g[g.xstest_type != focus])]:
                x = sub.assign(v=(sub.y == key).astype(int)).pivot(index="xstest_id", columns="persona_id", values="v")[PERSONAS].values
                rows.append({"scenario": scn, "outcome": "under-refusal (complied)" if stratum == "unsafe" else "over-refusal (refused)",
                             "prompt_group": name, "n_prompts": len(x), "events_by_persona": ",".join(str(int(v)) for v in x.sum(0)),
                             "Q_p": cochrans_q(x).pvalue})
    return pd.DataFrame(rows)


def misclass_sim(d, B, sims, rng) -> pd.DataFrame:
    """Random label flips in the unreviewed large cells (primary scenario)."""
    y0 = scenario_labels(d, "primary")
    unrev = (d.label_source == "judge")
    a = (unrev & (d.judge2 == "full_compliance") & (d.safety_label == "safe")).values   # compliance|safe -> may be refusal
    b = (unrev & (d.judge2 == "full_refusal") & (d.safety_label == "unsafe")).values    # refusal|unsafe -> may be compliance
    rows = []
    for e in [0.0, 0.02, 0.05, 0.08]:
        hits = {"over Q": 0, "under Q": 0, "H2 J Wald": 0}; n = 1 if e == 0 else sims
        for _ in range(n):
            y = y0.copy().values
            y[a & (rng.random(len(d)) < e)] = "R"
            y[b & (rng.random(len(d)) < e)] = "C"
            m = matrices(d, pd.Series(y))
            est = point_stats(m); bt = bootstrap(m, B, rng)
            hits["over Q"] += cochrans_q(m["safe"]["R"]).pvalue < 0.05
            hits["under Q"] += cochrans_q(m["unsafe"]["C"]).pvalue < 0.05
            hits["H2 J Wald"] += wald_boot(est["J"], bt["J"])[1] < 0.05
        rows.append({"flip_rate": e, "sims": n, **{k + " p<0.05": v / n for k, v in hits.items()}})
    return pd.DataFrame(rows)


def figures(rates: pd.DataFrame):
    names = {"neutral": "neutral", "childrens_tutor": "tutor", "medical_professional": "medical",
             "cybersecurity_professional": "cyber", "creative_fiction_writer": "creative", "customer_support_agent": "support"}
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for a, (col, title) in zip(ax, [("over_refusal", "Over-refusal (safe prompts refused)"), ("under_refusal", "Under-refusal (unsafe prompts complied with)"), ("discrimination", "Discrimination (refusal unsafe - refusal safe)")]):
        x = np.arange(len(PERSONAS))
        pr = rates[rates.scenario == "primary"].set_index("persona").loc[PERSONAS]
        a.errorbar(x, pr[col] * 100, yerr=[(pr[col] - pr[col + "_lo"]) * 100, (pr[col + "_hi"] - pr[col]) * 100], fmt="o", color="black", capsize=3, label="primary (95% CI)")
        for scn, mk, off in [("strict", "v", -0.15), ("lenient", "^", 0.15), ("judge_only", "x", 0.3)]:
            s = rates[rates.scenario == scn].set_index("persona").loc[PERSONAS]
            a.scatter(x + off, s[col] * 100, marker=mk, label=scn, s=40)
        a.set_xticks(x); a.set_xticklabels([names[p] for p in PERSONAS], rotation=30); a.set_title(title, fontsize=10); a.set_ylabel("%")
    ax[0].legend(fontsize=8)
    plt.tight_layout(); fig.savefig(OUT / "fig_rates.png", dpi=160); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--firth-boot", type=int, default=1000)
    ap.add_argument("--sims", type=int, default=200)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    d = load()
    rts, tss, cfs = [], [], []
    for scn in SCENARIOS:
        print("scenario", scn, flush=True)
        rt, ts, cf, m, est = run_scenario(d, scn, args.boot, args.firth_boot, rng)
        rts.append(rt); tss.append(ts); cfs.append(cf)
        if scn == "primary":
            type_rates(d, scenario_labels(d, "primary")).to_csv(OUT / "type_rates_primary.csv", index=False)
    rates = pd.concat(rts); tests = pd.concat(tss); coefs = pd.concat(cfs)
    rates.to_csv(OUT / "rates_by_scenario.csv", index=False)
    tests.to_csv(OUT / "tests_by_scenario.csv", index=False)
    coefs.to_csv(OUT / "firth_coefficients.csv", index=False)
    figures(rates)
    type_tests(d).to_csv(OUT / "type_tests.csv", index=False)
    print("misclassification simulation", flush=True)
    misclass_sim(d, 200, args.sims, rng).to_csv(OUT / "misclassification_sim.csv", index=False)
    print("done; outputs in", OUT)


if __name__ == "__main__":
    main()
