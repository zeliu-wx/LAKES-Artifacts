# LAKES

> A deterministic, fully traceable scheduler for smart-contract vulnerability detectors.

Given one Solidity contract, LAKES automatically decides which of 20 detectors (Slither, Mythril,
Confuzzius, VulHunter, Mando-HGT, ...) to run, **why** each one is chosen, and **which vulnerability
categories still cannot be covered**. Every recommendation comes with evidence—either metrics
computed inside the pipeline, or passages retrieved from a paper knowledge base—and must pass 10
checks before any tool is actually run.

> The system is named **LAKES**. Its Python package and CLI entry point are still `toolrank`, so the
> commands below are invoked as `toolrank ...`.

## Why

- Running all 20 tools together wastes compute, produces conflicting reports, and leaves you unable
  to tell which one is right.
- Running just one misses whole categories of bugs (Slither is weak on `bad_randomness`; Confuzzius
  covers less than 5% of `access_control`).
- Picking by experience is not reproducible, not auditable, and impossible for a newcomer to take over.

LAKES turns tool selection into an **explicit, traceable pipeline**: every step has a name, defined
inputs and outputs, and a list of citations.

## Architecture

```
                       contract.sol
                            │
                            ▼
  [1] Scene Match       ─→  match the 5 most similar benchmark scenes
  [2] Diagnostics       ─→  recall coverage / certification / ownership panel
  [3] Evidence Packet   ─→  primary tool + weak-category partition + DACE retrieval targets
  [4] DACE-RAG          ─→  enumerate 3 candidate actions × 4 evidence slots
                            (FOR / AGAINST / COMPARE / GAP), fill in internal
                            evidence + retrieve ~27 paper passages
  [5] CEGO              ─→  an LLM selects one action under 26 constraint rules
  [6] Checker           ─→  10 sub-checks; on failure → return to CEGO to rewrite
  [7] Execute + Fuse    ─→  run the selected tool combination and merge the reports
                            │
                            ▼
                    fused_report.json (with a complete audit chain)
```

**Every stage except CEGO is deterministic and reproducible.** CEGO's output must pass all 10 Checker
rules before it can enter execution—if the LLM gets it wrong, nothing runs.

## Quick start

```bash
python -m pip install -e ".[dev]"

# Configure your own OpenAI-compatible endpoints (supply the URLs and keys yourself)
export OPENAI_API_KEY=...              # key for the chat LLM
export TOOLRANK_OPENAI_BASE_URL=...    # base URL for the chat LLM
export TOOLRANK_EMBEDDING_API_KEY=...  # key for the embedding endpoint
export TOOLRANK_EMBEDDING_BASE_URL=... # base URL for the embedding endpoint

# One command, end to end (recommend + execute + fuse)
toolrank recommend path/to/Contract.sol --execute --emit summary
```

The fused report is written to `LAKES_out/<contract-name>/fused_report.json`. Add `-x` / `--explain`
to print the intermediate state of every stage—useful for demos and inspection.

## CLI

```bash
toolrank recommend <contract.sol> [options]   # main pipeline
  --execute            also run the selected tools and merge their reports
  --emit summary|json  terminal output format
  -x, --explain        print the details of every stage
  --no-retrieval       disable RAG retrieval (ablation)
  --jobs N             tool parallelism (default: all cores)

toolrank kb-extract <papers-dir>    # extract the scheduling knowledge base from a directory of papers
toolrank kb-audit <papers-dir>      # check knowledge-base integrity
toolrank kb-vector-build            # build the vector index for passage_store
toolrank refresh-kb                 # rebuild performance_db from raw reports
```

Run `toolrank <command> --help` for the full set of options.

## Core concepts

| Term | Meaning |
|---|---|
| **Scene** | A benchmark slice in the knowledge base, used to look up how tools performed historically on similar contracts |
| **R_hat** | The recall coverage of each (tool, vulnerability category) on historical data |
| **Confirmed-weak** | The primary tool has R_hat < 0.3 with a sample size ≥ 10 on the primary scene—"confirmed weak" |
| **DACE action** | One of three: `run_robust_single` / `plan_tool_composition` / `stop_with_gaps` |
| **FOR / AGAINST / COMPARE / GAP** | The four evidence slots of each action: support, opposition, head-to-head comparison, and known gaps |
| **Passage** | Paragraph-level evidence from a paper, tagged during KB extraction with structured labels such as `owner_tool`, `category`, `relation_to_owner` |
| **Ownership panel** | Which tool is responsible for each vulnerability category; if no suitable tool is found it is marked `gap` (an explicit admission that the category is unresolved) |

## Configuration

All endpoints are OpenAI-compatible, and **you supply the base URLs and keys yourself**—the repository
ships no provider addresses.

| Environment variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | API key for the chat LLM |
| `TOOLRANK_OPENAI_BASE_URL` | base URL for the chat LLM (defaults to local `http://127.0.0.1:8317/v1`) |
| `TOOLRANK_EMBEDDING_API_KEY` | API key for the embedding endpoint (falls back to `OPENAI_API_KEY`) |
| `TOOLRANK_EMBEDDING_BASE_URL` | base URL for the embedding endpoint (required, OpenAI-compatible) |
| `TOOLRANK_EMBEDDING_MODEL` | embedding model name (defaults to `Qwen/Qwen3-Embedding-8B`; the bundled index is built with it) |
| `TOOLRANK_SMARTBUGS_DIR` | explicitly set the SmartBugs location; otherwise auto-discovered |
| `TOOLRANK_RAG_STRICT_ERRORS` | set to `1` to make RAG retrieval failures raise instead of silently degrading |

Installation paths for the external analyzers (Securify2, GPTScan, Sailfish, Smartian) are likewise
set through `TOOLRANK_*` environment variables; see `toolrank/runner.py` for details.

## Docker: real execution

The image bundles LAKES, SmartBugs, and an internal Docker daemon (Docker-in-Docker), so you can run
the analyzers for real without installing any of them on the host:

- **SmartBugs-driven (16 tools):** slither, mythril, oyente, osiris, conkas, confuzzius, honeybadger,
  maian, manticore, sfuzz, smartcheck, solhint, securify, vandal, mando-hgt, vulhunter—each tool image
  is pulled automatically on first run.
- **Special-case analyzers:** securify2 (built inside the container against the contract's solc
  version), sailfish (public image), smartian (bundled .NET 8), gptscan (bundled venv, needs an LLM
  endpoint)—all packaged in the image.

Build:

```bash
docker build -t toolrank .
```

**Quickly verify real execution** (run a single tool, no LLM needed):

```bash
mkdir -p out
docker run --rm --privileged \
  -v toolrank-docker-cache:/var/lib/docker \
  -v "$PWD/out:/work/out" \
  --entrypoint bash toolrank -lc '
    dockerd >/var/log/dockerd.log 2>&1 & \
    for i in $(seq 1 60); do docker info >/dev/null 2>&1 && break; sleep 1; done; \
    python -m toolrank.runner examples/Reentrancy.sol /work/out \
      --tools slither --primary_tool slither --timeout 600'
# Result: out/LAKES_out/Reentrancy/raw/slither/result.json
```

**Full recommend + execute** (additionally needs your own LLM/embedding endpoints, used by CEGO and retrieval):

```bash
docker run --rm --privileged \
  -v toolrank-docker-cache:/var/lib/docker \
  -v "$PWD/out:/work/out" \
  -e OPENAI_API_KEY=... -e TOOLRANK_OPENAI_BASE_URL=... \
  -e TOOLRANK_EMBEDDING_API_KEY=... -e TOOLRANK_EMBEDDING_BASE_URL=... \
  toolrank recommend examples/Reentrancy.sol --execute --emit summary --results-root /work/out
```

Key points:

- **`--privileged` is required:** the image runs its own dockerd to launch the tool containers
  (Docker-in-Docker). This lets SmartBugs and the tool containers share one filesystem namespace,
  avoiding the path mismatches you get when mounting the host socket.
- **Image cache:** the named volume `-v toolrank-docker-cache:/var/lib/docker` persists pulled tool
  images (e.g. `smartbugs/slither:0.11.3`) across runs; the first run downloads them and is slower.
- **Cross-architecture:** some tool images are `linux/amd64`; on Apple Silicon they run under QEMU
  emulation—works, but slower.
- **securify2 is slow on first run:** it builds its analysis image inside the container against the
  contract's solc version (10+ minutes under emulation); with the persistent volume
  `-v toolrank-docker-cache:/var/lib/docker` it is reused afterwards.

## Audit trail

Every recommendation carries fields you can replay:

- `certification.reason_codes` — why the primary tool reached its current certification status
- `evidence_packet.dace_rag_focus` — which category each hedge tool fills, and which tier selected it
- `action.evidence[slot].refs` — each piece of evidence points to an internal evidence card (`ev_*`)
  or a paper passage (`p_*`)
- `checker.sub_checks` — the results of the 10 boolean sub-checks
- `category_decisions` — the final responsible tool for each vulnerability category, plus the list of
  supporting ref IDs

`ev_*` resolves to deterministic pipeline state; `p_*` resolves to a specific passage in
`toolcards/passage_store.json`. **The Checker rejects any decision that cites a ref ID not present in
the prompt**—the LLM cannot fabricate IDs.
