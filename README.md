# HR Intelligence

> AI-powered HR analytics agent — ask questions in plain English, get SQL + Python code + charts.

![Python](https://img.shields.io/badge/Python-3.10+-black?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-E82127?style=flat-square&logo=streamlit)
![Groq](https://img.shields.io/badge/LLM-Groq%20%2F%20OpenAI-white?style=flat-square)

---

## What it does

Type any HR question in plain English. The agent:

1. **Understands** your question (typos, follow-ups, vague phrasing — all handled)
2. **Generates SQL** and runs it on your HR dataset
3. **Generates Python code** (pandas + plotly) you can copy and reuse
4. **Renders up to 3 charts** automatically chosen for the data shape
5. **Writes a narrative insight** — like a smart colleague summarising the result

### Example questions

```
Show the recruitment funnel by stage
What is the offer-to-hire conversion rate?
Show monthly hiring trends as a line chart
Which department has the highest hire rate?
What % of applicants reach the Onsite stage?
Average time to hire in days?
Applications by quarter — bar chart
Top 10 most applied-to job positions
How many unique applicants per year?
```

---

## Repo structure

```
tesla-hr-intelligence/
│
├── app.py          ← Streamlit UI
├── agent_hr.py           ← AI agent core logic
├── Data_Challenge.csv    ← Your HR dataset
├── requirements.txt      ← Python dependencies
├── .streamlit/
│   └── secrets.toml      ← API key (local only, never commit this)
└── README.md
```

---

## Local setup

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/tesla-hr-intelligence.git
cd tesla-hr-intelligence
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your API key

Create the file `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "gsk_xxxxxxxxxxxxxxxxxxxx"
```

> Get a free Groq key at [console.groq.com](https://console.groq.com)
> Alternatively use `OPENAI_API_KEY` if you prefer OpenAI.

### 4. Run

```bash
streamlit run app_tesla.py
```

---

## Deploy on Streamlit Cloud (free)

1. Push this repo to GitHub (make sure `secrets.toml` is in `.gitignore`)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select your repo → set main file to `app_tesla.py`
4. Go to **Settings → Secrets** and paste:

```toml
GROQ_API_KEY = "gsk_xxxxxxxxxxxxxxxxxxxx"
```

5. Click **Deploy** — done ✓

---

## Data schema

The app loads `Data_Challenge.csv` and maps it to a SQLite table `hr_applications` with these columns:

| Column | Type | Description |
|---|---|---|
| `date_applied` | TEXT | ISO date (YYYY-MM-DD) |
| `applicant_id` | INTEGER | Unique applicant identifier |
| `stage` | TEXT | Application, Phone Screen, Onsite, Offer, Hired, Rejected |
| `job_level` | TEXT | e.g. S1 (I), M3 ( - ), S4 (IV) |
| `department_code` | INTEGER | Department identifier |
| `job_position_code` | INTEGER | Job position identifier |
| `target_start_date` | TEXT | ISO date or NULL |
| `year_applied` | INTEGER | Derived from date_applied |
| `month_applied` | INTEGER | Derived from date_applied |
| `quarter_applied` | TEXT | Q1 / Q2 / Q3 / Q4 |

You can also **upload additional CSVs** via the sidebar — the agent auto-detects join keys.

---

## Tech stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| LLM | Groq (`llama-3.3-70b-versatile`) or OpenAI (`gpt-4o`) |
| Database | SQLite (in-memory) |
| Charts | Plotly (bar, line, pie, funnel, scatter) |
| Data | Pandas |

---

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | Yes (or OpenAI) | Groq API key |
| `OPENAI_API_KEY` | Yes (or Groq) | OpenAI API key |

---

## .gitignore

Make sure your `.gitignore` includes:

```
.streamlit/secrets.toml
__pycache__/
*.pyc
.env
```
