"""
=====================================================================
 MPLADS AI Risk Intelligence Engine
 ---------------------------------------------------------------
 Shared core logic used by BOTH:
   1) The Google Colab notebook (training / batch scoring pipeline)
   2) The Streamlit dashboard (serving / display layer)

 IMPORTANT DESIGN PRINCIPLE
 ---------------------------------------------------------------
 This engine NEVER labels a project "fraudulent" or "corrupt".
 It only computes an explainable RISK / ANOMALY score that tells
 officials WHERE to look first and WHY, based on statistical
 deviation from normal MPLADS patterns. All outputs are meant to
 trigger human verification, not automated judgement.
=====================================================================
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

RUPEE = "₹"

# ---------------------------------------------------------------
# 0. LOADING & CLEANING
# ---------------------------------------------------------------

def load_raw_tables(completed_path, expenditure_path, mp_summary_path, recommended_path):
    """Load the four official MPLADS export CSVs and do light type cleaning."""
    completed = pd.read_csv(completed_path)
    expenditure = pd.read_csv(expenditure_path)
    mp_summary = pd.read_csv(mp_summary_path)
    recommended = pd.read_csv(recommended_path)

    # Strip whitespace from column names (defensive - some exports have trailing spaces)
    for df in (completed, expenditure, mp_summary, recommended):
        df.columns = [c.strip() for c in df.columns]

    # Parse dates
    completed["Completed Date"] = pd.to_datetime(completed["Completed Date"], errors="coerce")
    expenditure["Expenditure Date"] = pd.to_datetime(expenditure["Expenditure Date"], errors="coerce")
    recommended["Recommendation Date"] = pd.to_datetime(recommended["Recommendation Date"], errors="coerce")

    # Normalise MP name / constituency casing & whitespace to make joins reliable
    for df in (completed, expenditure, mp_summary, recommended):
        for col in ("MP Name", "Constituency", "State", "IDA"):
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()

    return completed, expenditure, mp_summary, recommended


# ---------------------------------------------------------------
# 1. VENDOR-LEVEL FEATURES  (built from the expenditure ledger)
# ---------------------------------------------------------------

def build_vendor_features(expenditure: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate the expenditure ledger to one row per MP, capturing vendor
    concentration and payment-pattern signals. Expenditure rows do not carry
    a Work ID (the 'Work Description' field there is a scheme *category*,
    not a specific project) so vendor analysis is done at MP level.
    """
    ex = expenditure.copy()
    ex["is_round_amount"] = (ex["Expenditure Amount (₹)"] % 10000 == 0)
    ex["is_success"] = ex["Payment Status"].eq("Payment Success")

    rows = []
    for mp, g in ex.groupby(["MP Name", "Constituency", "State", "House"]):
        total_amt = g["Expenditure Amount (₹)"].sum()
        vendor_amt = g.groupby("Vendor")["Expenditure Amount (₹)"].sum().sort_values(ascending=False)
        top_vendor = vendor_amt.index[0] if len(vendor_amt) else None
        top_vendor_share = (vendor_amt.iloc[0] / total_amt * 100) if total_amt > 0 and len(vendor_amt) else 0
        # Herfindahl-Hirschman style concentration index (0-100 scale, 100 = single vendor monopoly)
        shares = (vendor_amt / total_amt) if total_amt > 0 else vendor_amt * 0
        hhi = float((shares ** 2).sum() * 100)

        # same vendor, same day, multiple payments -> possible invoice splitting
        same_day_multi = (
            g.groupby(["Vendor", "Expenditure Date"]).size().gt(1).sum()
        )

        rows.append({
            "MP Name": mp[0], "Constituency": mp[1], "State": mp[2], "House": mp[3],
            "num_vendors": g["Vendor"].nunique(),
            "num_transactions": len(g),
            "total_expenditure_ledger": total_amt,
            "top_vendor": top_vendor,
            "top_vendor_share_pct": round(top_vendor_share, 2),
            "vendor_hhi": round(hhi, 2),
            "round_amount_txn_pct": round(ex_round := (g["is_round_amount"].mean() * 100), 2),
            "payment_inprogress_pct": round((1 - g["is_success"].mean()) * 100, 2),
            "same_day_same_vendor_clusters": int(same_day_multi),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------
# 2. COMPLETED-WORKS TRANSPARENCY FEATURES (aggregated to MP level)
# ---------------------------------------------------------------

def build_completed_features(completed: pd.DataFrame) -> pd.DataFrame:
    cw = completed.copy()
    cw["Category"] = cw["Category"].fillna("Unknown")

    # Cost outlier z-score within each work Category (peer-group comparison)
    cat_stats = cw.groupby("Category")["Final Amount (₹)"].agg(["mean", "std"]).rename(
        columns={"mean": "cat_mean", "std": "cat_std"})
    cw = cw.merge(cat_stats, on="Category", how="left")
    cw["cat_std"] = cw["cat_std"].replace(0, np.nan)
    cw["cost_zscore"] = (cw["Final Amount (₹)"] - cw["cat_mean"]) / cw["cat_std"]
    cw["cost_zscore"] = cw["cost_zscore"].fillna(0)

    # Duplicate / near-duplicate description reused by the same MP
    dup_counts = cw.groupby(["MP Name", "Work Description"]).size().rename("desc_repeat_count")
    cw = cw.merge(dup_counts, on=["MP Name", "Work Description"], how="left")

    # Exact same rupee amount repeated many times by the same MP (cookie-cutter billing)
    amt_counts = cw.groupby(["MP Name", "Final Amount (₹)"]).size().rename("amount_repeat_count")
    cw = cw.merge(amt_counts, on=["MP Name", "Final Amount (₹)"], how="left")

    cw["is_round_amount"] = (cw["Final Amount (₹)"] % 50000 == 0)
    cw["no_images"] = ~cw["Has Images"].astype(bool)
    cw["no_rating"] = cw["Average Rating"].isna()

    agg = cw.groupby(["MP Name", "Constituency", "State", "House"]).agg(
        completed_count=("Work ID", "count"),
        avg_cost_zscore=("cost_zscore", "mean"),
        high_cost_outlier_count=("cost_zscore", lambda s: int((s > 2).sum())),
        no_image_pct=("no_images", lambda s: round(s.mean() * 100, 2)),
        no_rating_pct=("no_rating", lambda s: round(s.mean() * 100, 2)),
        round_amount_pct=("is_round_amount", lambda s: round(s.mean() * 100, 2)),
        max_desc_repeat=("desc_repeat_count", "max"),
        max_amount_repeat=("amount_repeat_count", "max"),
    ).reset_index()

    return agg, cw  # return both MP-level rollup and enriched per-work table


# ---------------------------------------------------------------
# 3. MP-LEVEL RISK MODEL  (rule engine + Isolation Forest anomaly score)
# ---------------------------------------------------------------

RULE_WEIGHTS = {
    "utilization_completion_gap": 20,
    "vendor_monopoly": 18,
    "high_pending_payments": 12,
    "low_transparency_images": 10,
    "low_transparency_rating": 8,
    "high_round_amount": 8,
    "cost_outliers": 12,
    "duplicate_descriptions": 8,
    "repeated_exact_amounts": 8,
    "large_unpaid_balance": 10,
}


def score_mp_level(mp_summary, vendor_feats, completed_rollup):
    df = mp_summary.merge(vendor_feats, on=["MP Name", "Constituency", "State", "House"], how="left")
    df = df.merge(completed_rollup, on=["MP Name", "Constituency", "State", "House"], how="left")

    # Fill NA for MPs with no expenditure / no completed works yet
    fill_cols = ["num_vendors", "num_transactions", "top_vendor_share_pct", "vendor_hhi",
                 "round_amount_txn_pct", "payment_inprogress_pct", "same_day_same_vendor_clusters",
                 "completed_count", "avg_cost_zscore", "high_cost_outlier_count", "no_image_pct",
                 "no_rating_pct", "round_amount_pct", "max_desc_repeat", "max_amount_repeat"]
    for c in fill_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    df["pending_payment_ratio"] = np.where(
        df["Transaction Count"] > 0, df["Pending Payments"] / df["Transaction Count"] * 100, 0)

    df["util_completion_gap"] = (df["Utilization %"] - df["Completion Rate %"]).clip(lower=0)

    df["unpaid_balance_pct"] = np.where(
        df["Total Expenditure (₹)"] > 0,
        df["Balance Not Yet Paid to Vendors (₹)"] / df["Total Expenditure (₹)"] * 100, 0)

    # ---------------- Rule triggers (boolean) + reasons text -----------------
    reasons_col, checklist_col, score_col = [], [], []

    for _, r in df.iterrows():
        pts = 0
        reasons = []
        checklist = []

        if r["util_completion_gap"] >= 40:
            pts += RULE_WEIGHTS["utilization_completion_gap"]
            reasons.append(
                f"Funds utilization ({r['Utilization %']:.0f}%) is far ahead of physical completion "
                f"({r['Completion Rate %']:.0f}%) — a gap of {r['util_completion_gap']:.0f} points.")
            checklist.append("Verify that advance/interim payments correspond to actual physical progress on site.")

        if r["vendor_hhi"] >= 40 and r["num_transactions"] >= 5:
            pts += RULE_WEIGHTS["vendor_monopoly"]
            reasons.append(
                f"Vendor concentration is high (HHI={r['vendor_hhi']:.0f}); "
                f"top vendor '{r.get('top_vendor','-')}' handles {r['top_vendor_share_pct']:.0f}% of spend.")
            checklist.append("Check whether the dominant vendor was selected through fair/competitive process across multiple works.")

        if r["pending_payment_ratio"] >= 25 and r["Transaction Count"] >= 5:
            pts += RULE_WEIGHTS["high_pending_payments"]
            reasons.append(f"{r['pending_payment_ratio']:.0f}% of payment transactions are still pending.")
            checklist.append("Review pending-payment vendor invoices for delay reasons and documentation completeness.")

        if r["no_image_pct"] >= 60 and r["completed_count"] >= 3:
            pts += RULE_WEIGHTS["low_transparency_images"]
            reasons.append(f"{r['no_image_pct']:.0f}% of completed works have no geo/photo evidence uploaded.")
            checklist.append("Request site photographs / geo-tagged images for works missing visual evidence.")

        if r["no_rating_pct"] >= 90 and r["completed_count"] >= 3:
            pts += RULE_WEIGHTS["low_transparency_rating"]
            reasons.append("Almost none of the completed works have received citizen/beneficiary feedback ratings.")
            checklist.append("Cross-check beneficiary feedback mechanism / conduct spot citizen verification.")

        if r["round_amount_pct"] >= 50 and r["completed_count"] >= 3:
            pts += RULE_WEIGHTS["high_round_amount"]
            reasons.append(f"{r['round_amount_pct']:.0f}% of completed works are billed in suspiciously round figures.")
            checklist.append("Ask for itemised cost estimates/bills rather than lump-sum round amounts.")

        if r["high_cost_outlier_count"] >= 2:
            pts += RULE_WEIGHTS["cost_outliers"]
            reasons.append(
                f"{int(r['high_cost_outlier_count'])} completed works cost far above the average for their "
                f"work category (statistical outliers).")
            checklist.append("Compare quoted cost of outlier works with category benchmark / schedule of rates (SOR).")

        if r["max_desc_repeat"] >= 4:
            pts += RULE_WEIGHTS["duplicate_descriptions"]
            reasons.append(f"The same work description text is reused {int(r['max_desc_repeat'])} times.")
            checklist.append("Verify these are genuinely distinct works and not duplicate/split entries.")

        if r["max_amount_repeat"] >= 4:
            pts += RULE_WEIGHTS["repeated_exact_amounts"]
            reasons.append(f"The exact same rupee amount appears on {int(r['max_amount_repeat'])} separate completed works.")
            checklist.append("Check for templated/cookie-cutter billing rather than work-specific costing.")

        if r["unpaid_balance_pct"] >= 40 and r["Total Expenditure (₹)"] > 0:
            pts += RULE_WEIGHTS["large_unpaid_balance"]
            reasons.append(f"{r['unpaid_balance_pct']:.0f}% of expenditure is still an unpaid balance to vendors.")
            checklist.append("Confirm vendor payment schedule and reasons for outstanding dues.")

        score_col.append(pts)
        reasons_col.append(reasons)
        checklist_col.append(checklist)

    df["rule_score_raw"] = score_col
    df["reasons"] = reasons_col
    df["checklist"] = checklist_col

    # ---------------- Isolation Forest anomaly score (unsupervised) -----------------
    feature_cols = [
        "Utilization %", "Completion Rate %", "util_completion_gap", "vendor_hhi",
        "top_vendor_share_pct", "pending_payment_ratio", "no_image_pct", "no_rating_pct",
        "round_amount_pct", "avg_cost_zscore", "unpaid_balance_pct", "same_day_same_vendor_clusters",
    ]
    X = df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    scaler = MinMaxScaler()
    Xs = scaler.fit_transform(X)

    iso = IsolationForest(n_estimators=300, contamination=0.12, random_state=42)
    iso.fit(Xs)
    # decision_function: higher = more normal. Flip & scale to 0-100 (higher = more anomalous)
    raw_anomaly = -iso.decision_function(Xs)
    anomaly_scaled = MinMaxScaler((0, 100)).fit_transform(raw_anomaly.reshape(-1, 1)).ravel()
    df["ml_anomaly_score"] = anomaly_scaled.round(1)

    # ---------------- Hybrid final score -----------------
    rule_scaled = MinMaxScaler((0, 100)).fit_transform(df[["rule_score_raw"]]).ravel()
    df["rule_score_scaled"] = rule_scaled.round(1)
    df["risk_score"] = (0.6 * df["rule_score_scaled"] + 0.4 * df["ml_anomaly_score"]).round(1)

    def band(s):
        if s >= 70: return "High"
        if s >= 40: return "Medium"
        return "Low"
    df["risk_level"] = df["risk_score"].apply(band)

    df["reasons_text"] = df["reasons"].apply(lambda L: " | ".join(L) if L else "No significant rule triggered — flagged mainly by statistical pattern deviation." )
    df["checklist_text"] = df["checklist"].apply(lambda L: " | ".join(sorted(set(L))) if L else "General random-sample verification recommended.")

    return df.sort_values("risk_score", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------
# 4. WORK-LEVEL RISK MODEL (drill-down on individual completed works)
# ---------------------------------------------------------------

def score_work_level(cw_enriched: pd.DataFrame) -> pd.DataFrame:
    df = cw_enriched.copy()
    pts = np.zeros(len(df))
    reasons_list = [[] for _ in range(len(df))]

    cond = df["cost_zscore"] > 2
    pts += cond * 30
    for i in np.where(cond)[0]:
        reasons_list[i].append(
            f"Cost is a statistical outlier for its category (z-score={df['cost_zscore'].iloc[i]:.1f}).")

    cond = df["no_images"]
    pts += cond * 15
    for i in np.where(cond)[0]:
        reasons_list[i].append("No photographic/geo-tagged evidence uploaded for this work.")

    cond = df["no_rating"]
    pts += cond * 10
    for i in np.where(cond)[0]:
        reasons_list[i].append("No citizen/beneficiary rating recorded.")

    cond = df["desc_repeat_count"] >= 4
    pts += cond * 20
    for i in np.where(cond)[0]:
        reasons_list[i].append(
            f"Identical description reused {int(df['desc_repeat_count'].iloc[i])} times by this MP.")

    cond = df["amount_repeat_count"] >= 4
    pts += cond * 15
    for i in np.where(cond)[0]:
        reasons_list[i].append(
            f"Same exact amount seen on {int(df['amount_repeat_count'].iloc[i])} other works by this MP.")

    cond = df["is_round_amount"]
    pts += cond * 10
    for i in np.where(cond)[0]:
        reasons_list[i].append("Billed amount is a suspiciously round figure.")

    df["risk_points"] = pts
    df["risk_score"] = MinMaxScaler((0, 100)).fit_transform(df[["risk_points"]]).round(1)
    df["reasons_text"] = [" | ".join(r) if r else "No red flags — routine work." for r in reasons_list]

    def band(s):
        if s >= 70: return "High"
        if s >= 35: return "Medium"
        return "Low"
    df["risk_level"] = df["risk_score"].apply(band)
    return df.sort_values("risk_score", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------
# 5. END-TO-END PIPELINE
# ---------------------------------------------------------------

def run_full_pipeline(completed_path, expenditure_path, mp_summary_path, recommended_path):
    completed, expenditure, mp_summary, recommended = load_raw_tables(
        completed_path, expenditure_path, mp_summary_path, recommended_path)

    vendor_feats = build_vendor_features(expenditure)
    completed_rollup, cw_enriched = build_completed_features(completed)

    mp_risk = score_mp_level(mp_summary, vendor_feats, completed_rollup)
    work_risk = score_work_level(cw_enriched)

    return {
        "mp_risk": mp_risk,
        "work_risk": work_risk,
        "vendor_feats": vendor_feats,
        "raw": {
            "completed": completed, "expenditure": expenditure,
            "mp_summary": mp_summary, "recommended": recommended,
        }
    }
