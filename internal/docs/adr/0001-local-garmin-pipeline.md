# ADR-0001: Local Garmin tables and explicit RAG indexing

Date: 2026-11-09
Status: Accepted

## Context

The project contains a packaged Python skeleton and a full Garmin JSON export,
with detailed FIT files still in ZIP archives. The guide assumes root scripts,
unpacked FIT activities, and wellness CSVs. The user requested working modules
adapted to this project and commands usable from an activated virtual environment.

## Decision

Use small Python modules under `src/twil_demo`. Read existing JSON activity and
wellness summaries into two Parquet tables. Share deterministic calculations
between summaries, reports, and answers. Write weekly/monthly Markdown documents
and an explicit, persisted embedding index. Chat retrieves selected text chunks
and sends them to local Ollama; it does not rebuild the index implicitly. Keep the
static report independent of Ollama and offer Streamlit as an optional extra.

Data flow: Garmin JSON -> Parquet -> summaries/trends/charts -> HTML report.
RAG flow: summaries -> explicit embedding index -> retrieval -> local TwIL answer.

## Consequences

- Existing exports stay intact and FIT parsing is optional for future detail work.
- Data preparation and reporting work before any model is started.
- Missing values remain unknown; multisport parents are excluded when legs exist.
- RAG index freshness must be checked after summaries change.
- Numeric questions are limited to explicitly supported calculations and ranges.
- Shared paths assume a checkout; `TWIL_PROJECT_ROOT` supports a relocated data root.

## Alternatives considered

Root scripts match the guide literally but duplicate path/import handling.
A FIT-first importer adds archive processing and schema work when JSON summaries
already supply the required activity fields. A database/vector service adds a
daemon and dependencies unnecessary for a single local export.
