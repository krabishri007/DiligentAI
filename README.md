# MPLADS AI Risk Intelligence & Decision-Support Layer
### Smart India Hackathon Prototype

An **AI risk-detection and decision-support layer** built on top of official MPLADS
(Member of Parliament Local Area Development Scheme) data. It does **not** duplicate
the official MPLADS/eSAKSHI dashboard (https://mplads.mospi.gov.in/digigov/dashboard.html) — it sits
on top of it and answers three questions for officials:

1. **WHICH** MPs / works should be looked at first?
2. **WHY** do they look risky or anomalous?
3. **WHAT** should be verified first?

> ⚠️ **This system never claims fraud or corruption.** It only surfaces statistically
> anomalous patterns (vendor concentration, cost outliers, transparency gaps, etc.) that
> warrant human verification. Every flag ships with plain-language reasons and a
> verification checklist — the final judgement always stays with the official reviewing it.

---

## 📁 Project structure

```
mplads_risk/
├── MPLADS_AI_Risk_Colab.ipynb     ← PART 1: run this in Google Colab first
├── shared/
│   └── risk_engine.py             ← core ML + rule logic (used by both parts)
├── streamlit_app/                 ← PART 2: run this locally / deploy it
│   ├── app.py                     ← the dark-themed dashboard
│   ├── risk_engine.py             ← copy of the shared engine (self-contained app folder)
│   ├── requirements.txt
│   └── data/                      ← bundled pre-scored sample export (so the app
│       ├── mp_risk_scores.csv        works immediately without running Colab first)
│       ├── work_risk_scores.csv
│       ├── vendor_features.csv
│       └── meta.json
└── README.md                      ← you are here
```

The two parts are **fully separate and run independently**:

- The **Colab notebook** is the data-engineering / ML training pipeline: it takes the 4
  raw MPLADS export CSVs, engineers risk features, trains an Isolation Forest anomaly
  model, blends it with a rule engine, and exports 3 small risk-scored CSVs.
- The **Streamlit app** is the presentation / decision-support layer: it reads those
  exported CSVs (or lets an official upload fresh MPLADS CSVs and score them live) and
  renders the interactive dark-themed dashboard.

---

## ▶️ Part 1 — Run the Colab notebook

1. Open `MPLADS_AI_Risk_Colab.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Run cells top to bottom (`Runtime → Run all`).
3. When prompted in **Step 2**, upload the 4 official MPLADS export CSVs
   (completed works, expenditures, MP summary, recommended works).
4. At the end (**Step 7**), the notebook downloads `mp_risk_scores.csv`,
   `work_risk_scores.csv`, `vendor_features.csv`, and `meta.json`.
5. Copy those 4 files into `streamlit_app/data/`, replacing the bundled sample files.

*(The notebook can also be run outside Colab — e.g. Jupyter/VS Code — by setting the
4 file paths manually in Step 2.2 and skipping the `google.colab` upload cell.)*

---

## ▶️ Part 2 — Run the Streamlit dashboard locally

```bash
cd streamlit_app
python -m venv venv && source venv/bin/activate   # optional but recommended
pip install -r requirements.txt
streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`). The app works
immediately using the bundled sample data — no need to run the Colab notebook first
for a quick demo. You can also switch to **"Upload fresh MPLADS CSVs"** in the sidebar
to score a brand-new export live, right inside the dashboard.

---

## 🌐 Deploying the dashboard publicly (for judges / officials to access)

### Option A — Streamlit Community Cloud (free, easiest)
1. Push this whole project to a public/private GitHub repo.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub.
3. Click **New app**, pick your repo, set the main file path to `streamlit_app/app.py`.
4. Deploy — you'll get a public URL like `https://your-app.streamlit.app`.

### Option B — Hugging Face Spaces
1. Create a new Space → SDK: **Streamlit**.
2. Upload the contents of `streamlit_app/` (including `data/`) to the Space repo.
3. The Space auto-builds and gives you a public URL.

### Option C — Render / Railway / any Docker host
Use a minimal Dockerfile:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY streamlit_app/ /app
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

---

## 🧠 How the risk score works (summary)

**Hybrid model = 60% rule engine + 40% Isolation Forest anomaly detection**, blended
into a single 0–100 score per MP and per completed work, then banded into
**Low / Medium / High**.

Rule signals include: utilization-vs-completion gap, vendor concentration (HHI),
transparency gaps (missing photos/ratings), round-number billing, repeated exact
amounts, category-wise cost outliers, and large unpaid vendor balances.

See the **📘 Methodology** page inside the dashboard for the full explanation, or
`shared/risk_engine.py` for the exact code.

---

## Disclaimer

Built for Smart India Hackathon evaluation using publicly available MPLADS export
data. This is **not** an official Government of India product, and risk flags are
**leads for human verification**, not conclusions.
