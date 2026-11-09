# TwIL Garmin demo

Analyze the Garmin account export already in `data/garmin`, generate local reports,
and ask a local TwIL model questions grounded in computed facts and documents.

## Activate the Python environment

From this project directory:

```bash
uv sync
source .venv/bin/activate
```

After activation, use `python` or `twil-demo` directly. Run `deactivate` to leave
the environment. Activate it again in each new terminal. `uv sync` installs the
package in editable mode; code edits take effect without reinstalling. Run it
again after dependency changes. `uv run ...` remains an alternative to activation.

## Prepare data and generate the report

```bash
python -m twil_demo refresh
```

This runs ingestion, summaries, trends, charts, and the HTML report in order.
It does **not** call a model or build the RAG index. It is safe to repeat when the
export changes. Open `output/dashboard/report.html` in your browser.

The equivalent individual commands are:

```bash
python -m twil_demo.ingest
python -m twil_demo.summarize
python -m twil_demo.trends
python -m twil_demo.charts
python -m twil_demo.build_report
```

`twil-demo refresh` is the shorter installed command. Use `twil-demo --help` or,
for example, `twil-demo ingest --help` for available commands and arguments.

## Start the local models

The GGUF and `Modelfile` are already in this checkout. Start Ollama in another
terminal, or open the Ollama application:

```bash
ollama serve
```

In your project terminal, set up the named models once:

```bash
ollama create twil-8b -f Modelfile
ollama pull nomic-embed-text
ollama list
```

The embedding model download needs internet access once. With both models
available locally, indexing and answering use Ollama at `http://localhost:11434`.
Test direct model chat, which does not retrieve any Garmin data:

```bash
python -m twil_demo.chat "Explain RAG in one sentence."
# Or start an interactive direct conversation:
python -m twil_demo.chat
```

## Build the RAG index, then ask questions

First, explicitly embed the `.md` and `.txt` documents under `docs/`:

```bash
python -m twil_demo.rag_app index
```

This calls `nomic-embed-text` and saves text, source filenames, and vectors in
`output/rag_index.npz`. An unchanged index is reused without model calls. When
documents change, run `index` again; use `index --force` to rebuild regardless.
This implementation rebuilds the whole index when any document changes.

For exact table-backed questions:

```bash
python -m twil_demo.ask_garmin "How many hours did I run in the last 4 weeks?"
python -m twil_demo.ask_garmin "What was my average sleep in the last 7 days?"
python -m twil_demo.ask_garmin "How many bike sessions from 2026-10-01 to 2026-10-31?"
```

The deliberately small numeric router supports total/mean session duration,
total/mean distance, activity counts, and average sleep, resting HR, or weight.
Ask one metric at a time. Specify `last N days/weeks` or two ISO dates. Without a
range, numeric questions use the latest 28 calendar days in the export, not the
current computer date. Unsupported calculations and ambiguous ranges are rejected
instead of approximated. Returned table facts include dates and measurement
coverage. Counts include transition records; multisport parent totals are removed
when their child legs are present.

For document questions, use:

```bash
python -m twil_demo.rag_app ask "What observations appear in the latest training summaries?"
python -m twil_demo.rag_app chat
```

Answering embeds your question, retrieves four relevant text chunks, and sends
those excerpts plus your question to `twil-8b`. The Garmin command also computes
supported exact facts from the Parquet tables. TwIL does not receive the full
embedding index. Document RAG cannot reliably find a global maximum or total by
looking at four chunks: use computed metrics for arithmetic. Model citation and
grounding instructions are prompts, not a guarantee; check returned facts and
source documents. A missing or stale index gives a rebuild instruction.

## Optional interactive dashboard

```bash
uv sync --extra dashboard
source .venv/bin/activate
twil-demo dashboard
```

This starts Streamlit on localhost. Alternatively:

```bash
python -m streamlit run src/twil_demo/dashboard.py --server.address 127.0.0.1
```

Date and sport filters control activity views; date filters control wellness
views. The saved pattern observations and Ask panel describe the full export
independently of those filters, as labelled in the UI. The charts and tables work
without Ollama; the Ask button requires both Ollama and a current RAG index.
Use `uv sync --extra dashboard` when synchronizing to keep the optional packages.

## Goals and coach

On a fresh checkout, copy `goals.example.yaml` to `goals.yaml` once. Your personal
`goals.yaml` is ignored by Git; only the generic example is shared. Edit the local
file with your actual preferences and race dates, then run:

```bash
python -m twil_demo.coach
```

This explicitly calls TwIL with the saved `output/trends.json` and your goals.
The template assumes no race dates, training targets, or physiological thresholds.

## How this export is interpreted

- Activities: `DI_CONNECT/DI-Connect-Fitness/*summarizedActivities.json`, unwrapped
  from each `summarizedActivitiesExport` list. Distance is centimetres; duration
  is milliseconds. Deduplication uses `activityId`. Local start time determines
  the reporting date; UTC start time is retained separately.
- Daily wellness: `DI-Connect-Aggregator/UDSFile_*.json` provides resting HR and
  total-day stress. Missing measurements remain null on a complete daily calendar.
- Sleep: main sleep hours are deep + light + REM seconds divided by 3600. Awake
  time is excluded. Naps are stored separately as `nap_hours`. Overlapping main
  sleep records prefer confirmed-final, complete stage data, then latest end.
- Weight: nested dated measurements from `*userBioMetrics.json`, grams to pounds.
  Multiple measurements on a day use the latest available timestamp.
- VO2 max: `MetricsMaxMetData_*.json` retains running and cycling measurements
  separately; the compatibility `vo2max` column prefers running when both occur.
- Garmin readiness and native acute/chronic load metrics are also retained in
  the daily table. They are distinct from the report's calculated load ratio.
- The calculated load ratio is 7-day mean / 28-day mean of activity load. A
  recorded session without load makes that day's ratio inputs unknown; a
  transition without independent load counts as zero load. Days without sessions
  are treated as rest within the export's coverage. Windows with missing load
  stay unavailable. No missing-load estimates are invented.
- Seven-day chart averages require four observed measurements. Report cards
  display observed-measurement averages with coverage alongside them.
- ZIP archives remain intact. They are optional future sources for laps, GPS,
  HR time series, and other details, rather than required inputs for this build.

Read `data/clean/ingestion_report.json` for counts, coverage, and multisport
warnings after each refresh. Replace/update the raw export before refreshing;
this is a snapshot workflow, not a Garmin account connection.

## Files

```text
data/garmin/              Existing raw Garmin export
data/clean/               activities.parquet, daily.parquet, ingestion report
docs/garmin/              Generated weekly/monthly summaries, trends, patterns
models/                   Existing GGUF model
src/twil_demo/            Importable modules and command entry points
output/charts/            Four PNG charts
output/dashboard/         Static HTML report
output/trends.json        Computed facts for the coach
output/rag_index.npz      Persisted document index (after indexing)
goals.yaml                Private editable preferences (ignored)
goals.example.yaml        Generic goals template safe to share
tests/                    Synthetic data and mocked-model regression tests
```

Exports, model files, personal goals, credentials, and generated outputs are
excluded from Git. Ignore rules also block common data, archive, document, media,
and model extensions outside the usual folders. Code, Markdown documentation,
`goals.example.yaml`, and `uv.lock` remain eligible for version control. Review
`git status` before committing; ignore rules cannot detect private text pasted
into source files or prevent an explicit `git add -f`.

No model or
network requests occur when importing modules. Preparation/report commands make
no model calls. `TWIL_PROJECT_ROOT` can override the checkout root, and
`TWIL_OLLAMA_URL`, `TWIL_CHAT_MODEL`, `TWIL_EMBED_MODEL` configure the local client.
Changing the embedding model requires rebuilding the index.

## Verify

```bash
python -m unittest discover -s tests -v
```

Tests use synthetic data and mock Ollama calls; they do not require the model
server or send your Garmin data elsewhere.
