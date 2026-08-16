# LinkedIn Portfolio README Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite the repository README so LinkedIn visitors immediately understand the author's hybrid Data Engineering and Data Science/ML profile.

**Architecture:** Keep one English README with a portfolio-first opening and two explicitly independent workstreams: the lakehouse platform and financial ML research. Preserve runnable setup material below the portfolio summary, compress excessive implementation detail, and make only claims that can be verified in the repository.

**Tech Stack:** GitHub-flavoured Markdown, Mermaid, Docker Compose, Apache Airflow, Apache Kafka, Apache Spark, MinIO, PostgreSQL, Python, XGBoost, SHAP.

## Global Constraints

- Keep the primary README in English for broader reach from LinkedIn.
- Make the hybrid profile evident before the first architecture diagram.
- Do not imply that the financial module is orchestrated by Airflow or integrated into the lakehouse pipeline.
- Distinguish implemented functionality from scaffolding and roadmap items.
- Do not invent model performance figures, screenshots, production-readiness claims, or financial recommendations.
- Preserve a concise, reproducible Docker quick start and GitHub-compatible relative links.
- Modify documentation only; do not change application code or repository structure.

---

### Task 1: Portfolio-focused README

**Files:**
- Modify: `README.md`
- Reference: `docs/superpowers/specs/2026-08-16-linkedin-readme-design.md`
- Reference: `docker/docker-compose.yml`
- Reference: `airflow/dags/lakehouse_pipeline.py`
- Reference: `spark/spark_bronze.py`
- Reference: `models/get_data.py`
- Reference: `models/predictions.py`
- Reference: `models/compare_strategies_simple.py`

**Interfaces:**
- Consumes: repository source files and the approved README design.
- Produces: a self-contained GitHub landing page with valid internal links, Mermaid diagrams, setup commands, and accurate capability claims.

- [ ] **Step 1: Capture the pre-edit state**

Run:

```powershell
git diff -- README.md
git status --short
```

Expected: the existing user-authored README changes and untracked `LICENSE` are visible and remain outside destructive Git operations.

- [ ] **Step 2: Rewrite the README around the hybrid portfolio narrative**

Edit `README.md` to contain, in order:

1. An outcome-led title, concise hybrid positioning, badges, and project disclaimer.
2. `What this project demonstrates`, split into Data Engineering and Data Science/ML.
3. `Architecture and workflows`, with separate lakehouse and financial-research Mermaid diagrams.
4. `Implementation status`, explicitly marking Bronze as implemented and Silver/Gold as scaffolding.
5. Technology stack and repository map.
6. Prerequisites, quick start, endpoints, and useful Docker commands.
7. Key engineering decisions and a concise financial-ML methodology.
8. Limitations, roadmap, and MIT licence.

Retain these exact operational commands:

```bash
docker compose -f docker/docker-compose.yml --env-file .env up --build
docker compose -f docker/docker-compose.yml ps
docker compose -f docker/docker-compose.yml logs -f
docker compose -f docker/docker-compose.yml down
```

- [ ] **Step 3: Validate Markdown structure and repository links**

Run:

```powershell
$readme = Get-Content -Raw README.md
if (($readme.ToCharArray() | Where-Object { $_ -eq [char]96 }).Count % 6 -ne 0) { throw 'Review fenced code blocks manually' }
$links = [regex]::Matches($readme, '\[[^\]]+\]\((?!https?://|#)([^)]+)\)')
$missing = foreach ($link in $links) { $path = $link.Groups[1].Value.Split('#')[0]; if ($path -and -not (Test-Path -LiteralPath $path)) { $path } }
if ($missing) { throw "Missing README targets: $($missing -join ', ')" }
```

Expected: exit code 0 and no missing local targets. Then visually inspect that every Mermaid fence has a closing fence and each flow has a distinct heading.

- [ ] **Step 4: Validate claims against implementation**

Run:

```powershell
rg -n "lakehouse_pipeline|SparkSubmitOperator" airflow/dags/lakehouse_pipeline.py
rg -n "read\.format\(\"kafka\"\)|parquet|bronze|year|month|day|hour" spark/spark_bronze.py
rg -n "XGBClassifier|walk|SHAP|shap|HORIZON" models/predictions.py
rg -n "DCA|RSI|moving|value" models/compare_strategies_simple.py
```

Expected: each prominent README claim has corresponding source evidence; remove or qualify any claim without a match.

- [ ] **Step 5: Review the final documentation diff**

Run:

```powershell
git diff --check -- README.md
git diff --stat -- README.md
git diff -- README.md
git status --short
```

Expected: no whitespace errors, only the intended README rewrite plus the pre-existing untracked `LICENSE`, and no application-code changes.

- [ ] **Step 6: Commit only when explicitly requested**

Do not commit `README.md` or `LICENSE` automatically. If the user requests a commit, run:

```powershell
git add -- README.md LICENSE
git commit -m "docs: present project as hybrid data portfolio"
```
