# GridSFM — Full Capability Reference

This is a from-source reference of everything the `GridSFM` repository can
do: the `gridsfm` Python package's public API, exact function signatures,
the neural architecture, the checkpoint/caching machinery, the Julia
`topology_solver_pipeline`, and the browser viewer — including behavior
that the top-level and `model/` READMEs either abbreviate or omit
entirely. It was written by reading the source directly (`model/gridsfm/`,
`model/examples/`, `model/tests/`, `power_grid/US/topology_solver_pipeline/*.jl`,
`power_grid/US/viewer/`), not by summarizing the existing READMEs, so
where the two disagree this document should be treated as authoritative
for what the code actually does. Where the code is more capable than the
docs describe, or the docs describe something the code doesn't quite do,
that's called out explicitly.

The repo has two independent halves that only interact through a file
format (`.pyg.json`) and, informally, through the viewer:

- **`model/`** — a Python package (`pip install`-able as `gridsfm`) that
  loads a pretrained heterogeneous-graph-transformer AC-OPF surrogate and
  runs inference / fine-tuning.
- **`power_grid/US/`** — a self-contained Julia pipeline
  (`topology_solver_pipeline/`) that turns raw grid topology JSON into
  AC-OPF-solved `.pyg.json` scenarios, plus a stdlib-only Python HTTP
  server (`viewer/`) for browsing grid topology and OPF/ML results in a
  Leaflet map.

---

## 1. `model/` package layout and install

```
model/
├── gridsfm/            # the installable package
│   ├── __init__.py     # public API surface (see §2)
│   ├── model.py         GridTransformerBackbone (the neural net, §4)
│   ├── blocks.py         building blocks: attention, hetero-GNN, fusion
│   ├── signed_incidence.py  SignedIncidenceConv message-passing layer
│   ├── hodge_pe.py       Hodge-diffusion positional encoding
│   ├── cycle_basis.py     fundamental-cycle precompute + CycleBasisCache
│   ├── pe_features.py     DC-Laplacian PE features + LaplacianFactorizationCache
│   ├── dc_prior.py        DC-power-flow prior for the θ head + DCPriorCache
│   ├── stress_features.py physics-violation features feeding the feas head
│   ├── data.py            .pyg.json / OPFData loaders + batching
│   ├── schema.py          column-index constants (single source of truth)
│   ├── checkpoint.py      load_model / load_from_hf, v1.0→v1.1 adapter
│   ├── loss.py            compute_loss (fine-tuning, OPFData-only)
│   ├── eval.py            eval_pass (fine-tuning, OPFData-only)
│   ├── finetune_opfdata.py  finetune_opfdata training loop
│   ├── opfdata_train.py     OPFDataAdapterDataset (wraps PyG's OPFDataset)
│   ├── synthetic.py         SyntheticMixedDataset (4 infeasibility modes)
│   └── hf_util/gridsfm_pg_loader.py  GridSFM_PG_Loader (US power-grid HF loader)
├── samples/             # 53 unperturbed .pyg.json scenarios (§9)
├── examples/            # infer_samples.py, opfdata.py, predict_to_viewer.py,
│                        # finetune_opfdata_case6470.ipynb
├── tests/               # 6 pytest files, 38 tests total (§11)
├── checkpoints/         # gitignored local cache for downloaded ckpts
├── pyproject.toml
└── CHANGELOG.md
```

Install:

```bash
cd model && python -m venv .venv && source .venv/bin/activate
pip install -e .            # runtime deps only
pip install -e ".[test]"    # + pytest
pip install -e ".[notebook]" # + matplotlib, jupyter (for the FT example notebook)
```

Pinned deps (`pyproject.toml`): `torch>=2.6,<2.9`, `torch_geometric>=2.5,<3`,
`numpy>=1.24,<3`, `scipy>=1.10,<2`, `huggingface_hub>=0.34,<1`. Python
`>=3.10`, primary CI on 3.12. Package version is `gridsfm.__version__`
(currently `"1.1.0"`), read dynamically from `gridsfm/__init__.py` by
`pyproject.toml`'s `[tool.setuptools.dynamic]`.

`pyproject.toml` also carries a `[tool.uv.sources]` / `[tool.uv.index]`
pair (added in the most recent commit at the time of writing) that pins
`torch` to the `pytorch-cu128` wheel index
(`https://download.pytorch.org/whl/cu128`) when installed via `uv` — i.e.
`uv pip install -e .` / `uv sync` will pull a CUDA-12.8-linked torch build
specifically, not whatever the default PyPI torch wheel is. This has no
effect on a plain `pip install -e .`, which ignores `[tool.uv.*]` tables
entirely and resolves `torch` from PyPI's default index as normal.

Test invocation matters: the README explicitly warns to run
`python -m pytest`, not bare `pytest` — a user-local `~/.local/bin/pytest`
on `$PATH` can shadow the venv's interpreter and silently test against the
wrong install.

---

## 2. Public API (`gridsfm/__init__.py`)

Everything importable from the top-level `gridsfm` package:

```python
# Inference
GridTransformerBackbone, batch_data_list, load_model, load_from_hf,
load_pyg_json, load_opfdata, prepare_for_inference, predict, schema,
AC_LINE_KEY, TRANSFORMER_KEY,
# Fine-tuning (OPFData format only)
compute_loss, eval_pass, finetune_opfdata, OPFDataAdapterDataset,
SyntheticMixedDataset, __version__
```

### `predict(model, scenario, fmt="auto") -> Dict[str, Any]`

Single-graph inference only (raises `ValueError` if `scenario`'s batch
size is `> 1`; for batched inference, use `batch_data_list()` +
`model(batch)` directly — see §2.2). Decorated `@torch.no_grad()`.

- `scenario`: a `.pyg.json` / `.json` path, or an already-in-schema
  `HeteroData` (which is `.clone()`'d before mutation — the caller's
  object is never touched).
- `fmt`: `"auto"` (default; detects `.pyg.json` suffix → `"pyg"`, else
  `.json` → `"opfdata"`), `"pyg"`, or `"opfdata"`.

Returns a dict of **CPU** tensors/floats:

| key | shape | meaning |
|---|---|---|
| `theta` | `[n_bus]` | voltage angle, radians |
| `V` | `[n_bus]` | voltage magnitude, per-unit |
| `Pg`, `Qg` | `[n_gen]` | generator active/reactive power, per-unit |
| `Pij`,`Qij`,`Pji`,`Qji` | `[n_flow_edges]` | branch flows, concatenated across `ac_line` then `transformer` |
| `flow_edge_types` | `list[str]` | which families contributed (order matches concatenation) |
| `flow_edge_counts` | `list[int]` | row count per family — use to re-split the flat flow tensors |
| `feas` | `float` | `sigmoid(feas_logit)`, scenario feasibility probability in `[0,1]` |
| `feas_logit` | `float` | raw feasibility-head logit |

### 2.2 Batched inference: `model(batch)`

```python
prepared = [prepare_for_inference(load_pyg_json(p)) for p in paths]
batch = batch_data_list(prepared).to(device)
out = model(batch)   # HeteroData with predictions written onto it in place
```

`out["bus"].pred` is `[N_bus_total, 2] = (theta, V)`;
`out["generator"].pred` is `[N_gen_total, 2] = (Pg, Qg)`; per-edge-type
`out[("bus","ac_line","bus")].edge_flow_pred` / `...transformer...` are
each `[E, 4] = (Pij, Qij, Pji, Qji)`; `out.feas_logit` is `[n_graphs]`.
Use `torch_geometric.utils.unbatch(tensor, out["bus"].batch)` (or
`.batch` on the relevant node/edge store) to split per-graph.

`prepare_for_inference(data)` (in `data.py`) is **required** before any
forward pass and is **not** automatically called inside `predict()` for
already-loaded `HeteroData` objects passed by a caller who forgot it —
`predict()` does call it internally on whatever it loads, so the only
place a caller must remember it is the batched path. It: attaches the
cycle-basis + Hodge-PE node types (`branch_ac`, `branch_tr`, `cycle`),
widens `bus.x` from 4 → 16 columns (adds effective-resistance landmark
moments, DC-power-flow features, and a bus-type one-hot), and fills any
missing canonical node tensors with zero-row placeholders. It mutates
`data` in place and is idempotent (checked via `_gridsfm_cycle_attached`
/ `_gridsfm_pe_attached` sentinel attributes) — calling it twice is a
no-op the second time, not a double-transform.

### 2.3 Loaders (`data.py`)

- **`load_pyg_json(path) -> HeteroData`** — no schema validation beyond
  requiring a non-empty `bus` block and matching `senders`/`receivers`
  lengths per edge type; accepts any column width.
- **`load_opfdata(path) -> HeteroData`** — same envelope, but hard-enforces
  OPFData's exact column counts: bus=4, generator=11, `ac_line.features`=9,
  `transformer.features`=11. Raises `ValueError` naming the offending
  field on any mismatch. Both loaders require the top-level
  `{"grid": {"nodes": {...}}}` envelope — a bare `{"nodes": ...}` (no
  `"grid"` wrapper) is rejected by both, confirmed by
  `tests/test_opfdata.py::test_load_opfdata_rejects_flat_layout`.
- **`batch_data_list(data_list, copy=True) -> Batch`** — collates a list of
  **already-`prepare_for_inference`'d** HeteroData into a PyG `Batch`.
  Fills any canonical edge type missing from an individual graph with an
  empty tensor (so a mixed list — some grids with transformers, some
  without — batches cleanly), and stacks each graph's `g_ctx` (a per-graph
  8-dim context vector) into `batch.g_ctx`. Raises `ValueError` if any
  graph is missing `g_ctx` (i.e. wasn't run through `prepare_for_inference`
  first) — the error message lists the first 5 offending indices.
  `copy=True` (default) `.clone()`s every input first; the code comment
  notes `.clone()` is ~5000x faster than `copy.deepcopy()` on
  case6470-scale HeteroData because PyG's clone uses a tuned per-store
  tensor-clone path.

---

## 3. Input/output schema (`gridsfm/schema.py`)

`schema.py` is the single source of truth for column indices, imported by
essentially every other module (never hardcode an index elsewhere).

| node/edge type | columns (`.x` / `.edge_attr`) |
|---|---|
| `bus` | `[base_kV, type, Vmin, Vmax]` (4 raw cols; `prepare_for_inference` widens to 16 — see §2.2) |
| `generator` | `[mbase, _, Pmin, Pmax, _, Qmin, Qmax, Vg, cp2, cp1, cp0]` (11 cols; indices 1 and 4 unused by the model) |
| `load` | `[Pd, Qd, ...]` (≥2 cols) |
| `shunt` | `[Bs, Gs, ...]` (≥1 col; `Gs` optional) |
| `ac_line` edge_attr | `[angmin, angmax, b_fr, b_to, r, x, rate_a, ...]` (9 cols) |
| `transformer` edge_attr | `[angmin, angmax, r, x, rate_a, _, _, tap, shift, b_fr, b_to]` (11 cols) |

`BUS_TYPE_IDX` values follow MATPOWER/PowerModels convention: `1`=PQ,
`2`=PV, `3`=REF/slack (used throughout, e.g. `bus_type == 3` to find the
slack bus for the DC-prior solve and the θ anchor). `REACTANCE_FLOOR =
1e-4` is substituted for any `|x| < REACTANCE_EPS (1e-10)` before
inverting to admittance, guarding the Laplacian/DC-prior factorizations
against near-zero-reactance branches.

Every physical quantity is per-unit on the model's base MVA except
`base_kV` (nameplate kV) and `tap` (nameplate ratio); angles are radians.
This exactly mirrors the `.pyg.json` schema written by
`export_gridsfm_data.jl` on the Julia side (§13) — the two are meant to
be read together; `export_gridsfm_data.jl`'s header comment is the more
complete schema reference including the `solution` and `metadata` blocks
that the model package doesn't need but the viewer and
`predict_to_viewer.py` do.

---

## 4. Model architecture (`GridTransformerBackbone`, `model.py`)

This is a **heterogeneous graph transformer**, not a plain GNN or a plain
transformer — every block mixes per-node-type linear self-attention with
heterogeneous graph convolution. Default hyperparameters (all overridable
via the constructor, but the released checkpoints were trained with these
exact defaults and `load_model` will raise if you construct a mismatched
architecture and try to load release weights into it):

```python
GridTransformerBackbone(
    hidden_dim=128, num_blocks=8, num_heads=4, ffn_mult=4, dropout=0.0,
    leaky_alpha=0.0, input_norm=True, signed_fusion=True,
    hodge_pe_dim=8, hodge_pe_steps=4, theta_res_scale=1.0,
)
```

The released `gridsfm_open_v1.1.pt` has **15,148,227 parameters** total
(asserted by `tests/test_ft.py::test_load_v1_1`).

**Node/edge type system** — 7 node types (`bus`, `generator`, `load`,
`shunt`, `branch_ac`, `branch_tr`, `cycle`) and 14 directed edge-type
relations, including two "cell-complex"-style relations that don't exist
in the raw input and are synthesized by preprocessing:
- `branch_ac` / `branch_tr` are **branches promoted to nodes**
  (`cycle_basis.py:_promote_to_branch_nodes`), carrying the raw edge
  features as node features, connected to their two endpoint buses via
  signed `endpoint_of` edges (sign `+1`/`-1` by bus-index order).
- `cycle` nodes are the **fundamental cycle basis** of the branch graph
  (`cycle_basis.py:_fundamental_cycles_fast`, computed via a
  minimum-spanning-tree + non-tree-edge fundamental-cycle construction),
  connected to their member branches via signed `in_cycle` edges. This
  gives the model an explicit representation of Kirchhoff's Voltage Law
  loop structure.

**Per-block structure** (`blocks.py:GridBlock`, applied 8× by default):
1. Pre-LN **linear self-attention** per node type (`LinearSelfAttention`
   — Performer-style: `elu(x)+1` feature map, O(N) per-graph attention via
   cumulative `K^T V` and `K` sums scattered by graph index, not
   quadratic in node count).
2. Pre-LN **heterogeneous graph convolution** (`HeteroConv` over all 14
   edge types): 6 "topology" relations
   (`endpoint_of`/`in_cycle` pairs — cycle↔branch, bus↔branch) use a
   custom **`SignedIncidenceConv`** (`signed_incidence.py`) that
   multiplies the message by `edge_attr[:, 0]` (the ±1 orientation sign)
   before mean-aggregating — a physically-motivated choice since these
   edges represent an oriented incidence structure, not an undirected
   adjacency. The remaining relations (generator/load/shunt links) use a
   plain `SAGEConv`. Edges with zero count for a graph in a batch are
   masked to exactly zero output post-hoc (`_patch_conv_mask_no_edge_graph`)
   rather than relying on the conv's own empty-input handling.
3. Pre-LN **FFN** (`Linear → GELU → Linear`, width `ffn_mult × hidden_dim`).

**Hodge positional encoding** (`hodge_pe.py:HodgePE`) — before the main
blocks, a learned signed-Hodge-Laplacian diffusion process runs `T=4`
steps over each of the `bus` (0-form), `branch_ac`/`branch_tr` (1-form),
and `cycle` (2-form) node types, producing an 8-dim PE per node that is
concatenated onto the raw features before the first block's input
projection. This is conceptually a discrete-exterior-calculus /
cellular-sheaf positional encoding, not a standard random-walk or
Laplacian-eigenvector PE.

**Fusion + heads** (`blocks.py:FusionLayer`, `model.py:_predict_and_write`):
after the blocks, per-bus and per-generator embeddings are fused with
aggregated neighbor context (from ac lines, transformers, generators,
loads, shunts, and cycles — the last via a two-hop cycle→branch→bus
scatter), and a per-graph **global readout** concatenates **mean and max**
pooling over 4 node-type buckets (bus / branch_ac / branch_tr / cycle) into
an 8·d-wide vector before a final `Linear(8d, d)` (`W_global`) — this
mean+max pooling is the v1.1 architecture; v1.0 pooled mean only (see §5).
Five 2-layer MLP heads (`Linear → GELU → Linear`) produce:
- `head_theta`, `head_V` — clipped to `[Vmin, Vmax]` / unconstrained-then-
  anchored respectively (θ is anchored to the slack-bus mean via
  `_slack_or_mean_anchor`, and additionally biased by a **DC power-flow
  prior** — see below).
- `head_Pg`, `head_Qg` — clipped to `[Pmin, Pmax]` / `[Qmin, Qmax]` via a
  differentiable "leaky clip" (`leaky_alpha=0.0` by default → hard clip).
- `head_feas` — takes the global embedding + a per-graph `g_ctx` context
  vector + a 24-dim **physics-violation stress vector**
  (`stress_features.py`, computed from the *already-produced* θ/V/Pg/Qg
  predictions: voltage-bound, KCL-P, KCL-Q, thermal-loading, angle, and
  KVL-residual violations, plus 6 utilization/capacity-margin features)
  and outputs a single feasibility logit.

**DC power-flow prior** (`dc_prior.py`) — θ is not predicted from scratch;
the model computes a classical DC power-flow solution (`compute_dc_prior`)
from the predicted `Pg` and adds it as a `tanh`-squashed bias before the
learned residual, giving the network a physically-sensible starting point.
This uses a cached SuperLU factorization of the (topology, susceptance,
slack-bus)-keyed reduced admittance Laplacian, batched per-graph, with a
generation/load rescaling clamp (`DC_PRIOR_SCALE_MIN=0.5` /
`DC_PRIOR_SCALE_MAX=2.0`) to keep the prior stable when predicted `Pg`
hasn't converged to balance load yet.

**Branch flows are not a learned head** — after θ/V are produced,
`_compute_flows` computes `Pij/Qij/Pji/Qji` **analytically** via the
standard π-model transmission-line/transformer equations
(`_pi_model_flows_ac_line`, `_pi_model_flows_transformer`; the latter
handles off-nominal tap and phase shift). This is deterministic
post-processing, not a neural prediction — verified in
`tests/test_pi_model.py` (lossless-line reciprocity, resistive loss sign,
transformer reducing to ac_line at unit tap / zero shift, radial-network
power balance).

---

## 5. Checkpoints (`checkpoint.py`, `hf_util/`)

There are exactly **two** released checkpoints, both under one HF repo
(`microsoft/GridSFM_Open`, part of the `microsoft/gridsfm` collection) —
there is no larger registry of model-size variants; "which model to use"
is a two-way choice, not a matrix:

| ckpt | status | backbone difference |
|---|---|---|
| `gridsfm_open_v1.1.pt` | recommended, required for fine-tuning | fusion pools mean **and** max per node-type bucket; `W_global = Linear(8d, d)` |
| `gridsfm_open_v1.0.pt` | deprecated, inference-only | fusion pools mean only; `W_global = Linear(4d, d)` |

```python
from gridsfm import load_from_hf, load_model
model = load_from_hf("microsoft/GridSFM_Open")   # default filename gridsfm_open_v1.1.pt
# or:
model = load_model("checkpoints/gridsfm_open_v1.1.pt", device="cuda:0")
```

`load_from_hf(repo_id, filename="gridsfm_open_v1.1.pt", revision=None,
cache_dir=None, **kwargs)` is a thin wrapper: `hf_hub_download` then
`load_model(..., **kwargs)`; `kwargs` (e.g. `device=`) pass through.

`load_model` does more than a `torch.load`:
1. Loads with `weights_only=True`; requires top-level keys `state_dict`
   and `metadata` (training-format checkpoints with `model_state_dict` /
   `optimizer_state_dict` are explicitly rejected with a message pointing
   at "the export script in the training repo").
2. Recomputes a SHA-256 over the sorted state-dict (`_hash_state_dict`)
   and compares against `metadata["hash"]`; mismatch raises `ValueError`
   naming both hashes (catches partial/corrupted downloads and in-place
   file tampering). Missing `metadata.hash` also raises.
3. Builds `GridTransformerBackbone(**metadata.get("arch", {}))` — so a
   checkpoint carrying a non-default architecture in its metadata builds
   the matching model automatically.
4. Per-key shape check: exact match loads as-is; the **one** known
   cross-version case (`fusion.W_global.weight`, v1.0 4d-wide →  v1.1
   8d-wide) is permuted via `_adapt_v1_0_w_global` — v1.0's 4 per-type
   `d`-wide mean columns land in the v1.1 layout's even (mean) slots;
   the odd (max) slots are **explicitly zeroed**, not left at Kaiming
   init, so the mean pathway dominates at load time (`CHANGELOG.md`
   confirms this was a deliberate choice, verified in
   `tests/test_ft.py::test_load_v1_0_cross_version_adapt`). Any other
   shape mismatch, any ckpt key absent from the model, or any model
   parameter absent from the ckpt raises `RuntimeError` rather than
   silently zero-padding/cropping/discarding/random-initializing — the
   stated rationale (in both `checkpoint.py` and `CHANGELOG.md`) is that
   any of those would mask an architecture mismatch that slipped past
   the hash check.
5. A v1.0 load emits `DeprecationWarning` naming the `hf download` command
   to fetch v1.1.
6. `model.to(device).eval()`.

---

## 6. Caches, and what "large N-1 sweeps" actually means

Three independent LRU caches, each keyed by a **SHA-1 hash of topology
structure** (`pe_features.py:hash_topology` / `_topology_fingerprint`, or
`dc_prior.py:DCPriorCache.topo_key`) so that changing which lines are
present, which bus is slack, or an impedance value invalidates the cache
entry:

| cache | file | default size | keyed on | picklable? |
|---|---|---|---|---|
| `CycleBasisCache` | `cycle_basis.py` | in-memory LRU 128 | `n_bus` + per-edge-type `(edge_index, edge_attr)` bytes | yes — pure tensors; also has schema-versioned disk persistence at `$XDG_CACHE_HOME/gridsfm/cycle_basis/` (falls back to in-memory-only with a `RuntimeWarning` if the dir can't be created) |
| `LaplacianFactorizationCache` | `pe_features.py` | in-memory LRU 16 | topology + bus type + generator placement + edge attrs | **no** — holds SciPy `SuperLU` solve closures |
| `DCPriorCache` | `dc_prior.py` | in-memory LRU 64 | topology + `b=1/x` + slack bus index | **no** — same SuperLU-closure constraint |

`tests/test_cache_invalidation.py` confirms fingerprints change on
impedance edit, slack-bus reassignment, and generator-to-bus reassignment
(6 tests total, all fast unit tests with no checkpoint dependency).

**What this has to do with N-1 sweeps**: none of this is an N-1
*scenario generator* — it's purely a performance knob for whoever is
already producing many topology variants. If you run `predict()` (or
batched inference) across a large sweep of grids that share a base
topology but differ by which lines/generators are in service (e.g. a
manually-constructed N-1 or N-k contingency set), each distinct topology
pays a fresh cycle-basis computation and two sparse factorizations unless
its hash is already cached — and the small default cache sizes (128/16/64)
will start evicting well before a few-thousand-topology sweep finishes
its first pass, forcing recomputation. The fix is to enlarge (and,
optionally, disk-persist) the caches **before first use**, since they're
module-level singletons read by default when no cache is passed
explicitly:

```python
import gridsfm.cycle_basis as cb, gridsfm.pe_features as pe
cb.DEFAULT_CYCLE_CACHE = cb.CycleBasisCache(max_cache=2048, cache_dir="/data/grid_cache")
pe.DEFAULT_LAPLACIAN_CACHE = pe.LaplacianFactorizationCache(max_cache=128)

model = load_model("checkpoints/gridsfm_open_v1.1.pt")
model._dc_cache = DCPriorCache(max_cache=512)   # per-model-instance, not global
```

**Notable gap**: the top-level README's `power_grid/` pipeline description
never mentions N-1/contingency generation, and rightly so — reading the
Julia scenario generator (§13) confirms it does **not** produce a
line-outage / N-1 perturbation mode at all. Its five modes are `loads`,
`costs`, `killgen` (generator, not line, outages), `derate`, and
`vsqueeze`. The only place genuine N-1 topology variation enters this
repo is the **external** OPFData dataset's `n1` split, reachable via
`OPFDataAdapterDataset(..., variant="n1")` (§7), which sets
`topological_perturbations=True` on PyG's own `OPFDataset` — i.e. N-1
data for fine-tuning/eval comes from OPFData, not from anything
`topology_solver_pipeline` generates. If a user wants to run inference
over a self-built N-1 sweep of their own `.pyg.json` grids, they must
construct those topology variants themselves (e.g. by editing
`grid.edges.ac_line.senders/receivers/features` and re-running
`prepare_for_inference` per variant) — the caching guidance above is
exactly for that use case, but nothing in the repo automates producing
the variants themselves outside of OPFData's pre-built n-1 split.

---

## 7. `gridsfm.hf_util.GridSFM_PG_Loader` — US power-grid dataset loader

Downloads/loads the `microsoft/GridSFM_US_power_grid` HF dataset (raw
PowerModels-compatible grid models + AC/DC OPF results), independent of
the model package's checkpoint loading. Metadata (region list, hour
labels, file-type list, abbreviation map, filename pattern) is **not**
hardcoded — it's fetched from `dataset_metadata.json` in the dataset repo
on first access and cached on the instance, so the loader tracks dataset
releases without code changes.

```python
loader = GridSFM_PG_Loader("microsoft/GridSFM_US_power_grid",
                            export_dir="./gridsfm_data")   # pre-fetches everything by default
model  = loader.load_model("texas", hour="16h")            # or "TX", "Texas", case-insensitive
ac     = loader.load_ac_results("texas", hour="16h")
bundle = loader.load_bundle("texas", hour="16h")            # {"model", "ac_results", "dc_results"}
loader.download_all("./data")                                # snapshot_download of the whole repo
```

`export_dir` + `pre_fetch_all=True` (default) triggers a full
`snapshot_download` in `__init__` — instantiating the loader with an
`export_dir` set is **not** lazy by default; pass `pre_fetch_all=False`
to defer. Region resolution (`_resolve_region`) accepts full names or
abbreviations, case-insensitively, via lookup tables built once from the
metadata. `summarize_model(model)` is a static helper that extracts
bus/branch/gen/load/shunt/dcline counts and total load MW from a loaded
model dict — useful for a quick sanity check without opening the file
in a viewer. Has a `__main__` CLI (`python -m gridsfm.hf_util.gridsfm_pg_loader
<repo_id> [--region ...] [--hour ...] [--download-all] [--list] [--export-dir ...] [--no-pre-fetch]`)
not mentioned in either README.

---

## 8. Fine-tuning (`loss.py`, `eval.py`, `finetune_opfdata.py`, `synthetic.py`, `opfdata_train.py`)

**Hard constraint, stated repeatedly in the code and CHANGELOG**:
fine-tuning is supported **only** on the v1.1 checkpoint, and **only** on
the OPFData dataset format (`torch_geometric.datasets.OPFDataset`). The
loss's column-index conventions (`_OPF_EDGE_SCHEMA` in `loss.py`) and
`SyntheticMixedDataset`'s perturbation logic are pinned to OPFData's
exact schema — arbitrary `.pyg.json` scenarios from `power_grid/` cannot
be fine-tuned against; they're inference-only via `predict()`/`model(batch)`.

### `OPFDataAdapterDataset(root, case_name, variant="fulltop"|"n1", split="train"|"val"|"test", n_graphs=None, num_groups=1, transform=None)`

Wraps PyG's `OPFDataset` (splits: train ~13.5k / val 750 / test 750
graphs). **Non-obvious cost**: `n_graphs` caps `__len__`/`__getitem__`
*after* the underlying `OPFDataset` finishes downloading and processing
its full `num_groups * 15000`-graph shard — so `n_graphs=16` does **not**
make the first construction cheap; only the *first* call pays the
download+decode cost (subsequent calls hit `_CachedOPFDataset`, which
skips re-download/re-process if `processed_paths` already exist).
`variant="n1"` sets `topological_perturbations=True` on the inner
`OPFDataset` — this is the repo's only source of N-1-varied topology data
(see §6's gap note).

### `SyntheticMixedDataset(base_dataset, infeas_prob=0.5, seed=42, transform=None, mode_weights=None)`

Wraps a feasible base dataset; with probability `infeas_prob` (default
0.5, the FT notebook uses 0.3) applies exactly one of 4 perturbation
modes (weights default `(0.25, 0.20, 0.20, 0.35)`, must sum to 1.0 or
`ValueError`) that turns a feasible graph infeasible, zeroing its
per-node-type `.y` labels and setting `feasible=0`:

| mode | what it does |
|---|---|
| `voltage_squeeze` | tighten `Vmin`/`Vmax` on 30–60% of buses + scale `Qd` ×1.2–1.8 |
| `thermal_bottleneck` | BFS-grow a corridor covering 10–25% of `ac_line` edges, derate their ratings ×0.05–0.20 (transformers touching the corridor ×0.10–0.30), plus per-load ±10–30% jitter |
| `angle_tighten` | tighten `angmin`/`angmax` on 20–40% of edges + scale load ×1.05–1.3 |
| `capacity_aware_spike` | pick 1–3 load buses (inverse-capacity-weighted toward weakly-connected buses), spike the largest load there by `target_ratio · Σrate_a` incident, `target_ratio` from a 3-band mixture in `[0.75, 5.0]` |

Deterministic per `(seed, epoch, idx)` — `set_epoch(ep)` must be called
each epoch to roll the perturbation seed forward; the docstring warns
that `DataLoader(persistent_workers=True)` breaks this (workers keep a
stale `_epoch`), so use `persistent_workers=False` if fresh perturbations
per epoch matter. `thermal_bottleneck` caches a BFS adjacency per unique
topology (memory scales with unique topology count — the docstring flags
`num_workers <= 2` as a guideline on a full n-1 split, ~13.5k topologies).
An import-time assertion checks every name in `_MODE_NAMES` has a
matching `_perturb_<name>` method, catching typos at load time rather
than via a mid-epoch `AttributeError`.

### `compute_loss(batch, lambda_feas=0.1, lambda_cost=1.0, lambda_stress_feas=0.1, lambda_kcl_p=1.0, lambda_kcl_q=1.0, lambda_br_p=1.0, lambda_br_q=1.0, lambda_therm=1.0, lambda_thermal_limit=5.0) -> (loss, parts_dict)`

13-component loss, single-GPU only (no DDP `find_unused_parameters`
plumbing): tanh-capped (`PER_ELEM_CAP=100.0`) squared error on θ (circular
distance via `atan2`) / V (per-graph-variance-normalized) / Pg / Qg, all
masked to feasible graphs; BCE on the feasibility logit (covers the full
batch, feasible and synth-infeasible alike); log-MSE on total generation
cost; a stress-feasibility regularizer (log1p of the 18 violation
dimensions from `compute_physics_stress`); differentiable KCL P/Q
residual (capacity-normalized, `0.8`-weighted blend of worst-bus and
mean-bus penalty per graph); supervised branch-flow P/Q MSE against the
solver's `edge_label`; supervised thermal-loading MSE; and a soft top-k
(`β=8.0` softmax-focus) thermal-limit barrier. `parts_dict` keys mirror
the lambda names; several (`L_kcl_p/q`, `L_br_p/q`) are **raw pre-`log1p`
values** — the actual loss contribution applies `log1p` at combine time,
so don't read `parts['L_kcl_p']` as directly comparable across runs
without accounting for that.

### `finetune_opfdata(model, train_loader, val_loader=None, epochs=10, lr=1e-4, weight_decay=1e-4, grad_clip=5.0, loss_kwargs=None, on_epoch_end=None) -> List[Dict]`

Standard AdamW + grad-clip loop; skips (with a warning) any batch whose
loss is non-finite rather than corrupting all parameters via
`opt.step()`; if literally zero batches contribute in an epoch,
`train_loss` is reported as `NaN` (not `0.0`, which would falsely read as
perfect convergence). Returns a per-epoch list of dicts
(`epoch, train_loss, n_train_iters, n_train_skipped, epoch_s, elapsed_s`,
plus `val_*` keys if `val_loader` given) suitable for direct loss-curve
plotting.

### `eval_pass(model, loader, device=None, loss_kwargs=None) -> Dict[str, Any]`

Returns `loss`, `cost_mape`, `pg_mae`, `qg_mae`, `V_mae`, `theta_mae`
(circular-distance), `brP_mae`, `brQ_mae`, `kcl_P_resid`, `kcl_Q_resid`,
`thermal_max_loading`, `thermal_frac_overload`, `feas_acc` (covers **all**
graphs, feasible and infeasible), `n_graphs`. Metrics are `NaN` (not 0)
when nothing contributes, for the same false-positive-avoidance reason as
`finetune_opfdata`.

### The worked example: `examples/finetune_opfdata_case6470.ipynb`

Few-shot FT study on `pglib_opf_case6470_rte`: 0-shot eval of the release
ckpt on `fulltop` + `n1` test splits, then three independent FT rounds
(`n_samples = 16 / 104 / 1000`, each starting fresh from release weights,
AdamW `lr=1e-4`, 10 epochs, `SyntheticMixedDataset(infeas_prob=0.3)`),
per-epoch loss-curve plots, and a final 0-shot-vs-FT comparison table.
End-to-end runtime on one GPU is roughly an hour, dominated by the n=1000
round (~3.4 min/epoch × 10 ≈ 35 min); n=16 finishes in a couple minutes
and works as a "is FT wired up correctly" smoke test.

---

## 9. `examples/` — what each script actually demonstrates

- **`infer_samples.py [ckpt_path] [--gpu N]`** — loads all 53 shipped
  samples, runs `prepare_for_inference` on each, batches them (mixed
  topology, one `torch_geometric.data.Batch`), runs one forward pass, and
  for every sample computes V/θ/Pg/Qg MAE plus generation-cost MAPE
  against the ground-truth `solution` block bundled in each sample file,
  reporting a per-case table and aggregate feasibility-classifier
  accuracy. This is the fastest way to sanity-check a checkpoint end to
  end without any external dataset dependency.
- **`opfdata.py --case <pglib_case> [--split train|val|test] [--root DIR]
  [--batch-size 128] [--limit N] [--num-groups N] [--ckpt PATH] [--gpu N]`**
  — the one script that actually exercises the OPFData benchmark format
  (arXiv:2406.07234) end to end: instantiates
  `torch_geometric.datasets.OPFDataset` (auto-downloads on first use),
  iterates the full split in chunks, and reports aggregate V/θ/Pg/Qg MAE
  and cost MAPE. Explicitly warns that cases below 500 buses
  (`case14_ieee`, `case30_ieee`, `case57_ieee`, `case118_ieee`) are out of
  distribution for GridSFM-Open and results there are not meaningful.
- **`predict_to_viewer.py --sample PATH --out PATH [--ckpt PATH] [--gpu N]`**
  — **not mentioned in either README**, but it's the bridge between the
  model package and the viewer (§14): runs `predict()` on one
  `.pyg.json` sample and writes a `*_gridsfm_results.json` in the exact
  schema the viewer's DC/AC results files use (same `bus`/`gen`/`branch`
  solution dict shape, `baseMVA`, `objective`, `termination_status` etc.),
  so the viewer can render an ML-predicted dispatch alongside a
  solver-computed one for the same grid. Requires the sample's `metadata`
  block to carry `bus_id_map`/`gen_id_map`/`ac_line_branch_ids`/
  `transformer_branch_ids` (present in every shipped `.pyg.json` sample
  and every scenario `export_gridsfm_data.jl` produces) to map row
  indices back to original PowerModels IDs; raises loudly on any
  mismatch between `flow_edge_types`/`flow_edge_counts` and the expected
  branch-id counts rather than silently mis-mapping flows.

---

## 10. `samples/` — the 53 shipped scenarios

23 `case*` pglib-opf base cases (`case500_goc` through `case3375wp_k`) +
30 `msr_*` snapshots (16h peak-demand) derived from the
`microsoft/GridSFM_US_power_grid` HF dataset, covering `msr_arizona`
through `msr_wisconsin` (mostly individual US states, plus multi-state
regions `msr_desert_sw`, `msr_new_england`, `msr_pacific_nw`). All 53 are
≥500 buses, consistent with the "GridSFM-Open is trained on ≥500-bus
grids" constraint stated in the model README. Every file bundles a
`solution` block (the AC-OPF dispatch from the data-generation solver),
so samples double as both inference inputs and ground-truth comparison
targets (used by `infer_samples.py`).

**Explicitly base cases only** — none of the five `topology_solver_pipeline`
perturbation modes (loads, costs, killgen, derate, vsqueeze — §13) are
applied here. To get perturbed `.pyg.json` scenarios for inference, run
the Julia pipeline.

---

## 11. `tests/` — what's actually covered

6 files, 38 tests total (verified via `python -m pytest --collect-only -q`:
`test_cache_invalidation.py` 6, `test_ft.py` 13, `test_opfdata.py` 6,
`test_pi_model.py` 6, `test_smoke.py` 4, `test_stress.py` 3). Tests requiring a checkpoint
(`gridsfm_open_v1.1.pt` under `model/checkpoints/`) `pytest.skip()`
gracefully if it's absent rather than failing — so a fresh clone with no
downloaded weights still gets useful signal from the checkpoint-free
tests. `GRIDSFM_TEST_DEVICE` env var (default `"cpu"`) selects the device
for checkpoint-dependent tests.

| file | covers |
|---|---|
| `test_smoke.py` | `predict()` output shapes/finiteness on a real sample; single-vs-batched forward numerically agree (`atol/rtol=2e-3`, explicitly wider than a typical unit-test tolerance to absorb GPU FP32 reduction-order nondeterminism in scatter-based ops); `predict()` rejects a multi-graph batch; flow-edge-type/count bookkeeping is self-consistent |
| `test_opfdata.py` | `load_opfdata` column-width enforcement (rejects too-narrow/too-wide bus rows, wrong-width `ac_line.features`, missing `"grid"` wrapper); a round-trip check that `predict()` on the same scenario loaded via `fmt="pyg"` vs `fmt="opfdata"` agrees |
| `test_pi_model.py` | the analytic π-model flow equations directly (zero flow at zero angle diff + zero R/X, resistive-loss sign, lossless reciprocity `Pij≈-Pji`, transformer-equals-ac-line at unit tap/zero shift, a 5-bus radial chain's flow balance) — **unit tests of physics, not of the learned model** |
| `test_stress.py` | `compute_physics_stress` output shape (`[G, 24]`), finiteness, and that 3 identical stacked graphs in one batch produce bit-identical stress rows (deterministic per-graph pooling) |
| `test_cache_invalidation.py` | all three cache fingerprints (§6) change under impedance edit, slack reassignment, generator-bus reassignment |
| `test_ft.py` | v1.1 param count (15,148,227) and `W_global` shape assertion; v1.0→v1.1 cross-version load emits `DeprecationWarning` and correctly permutes/zeroes `W_global` columns; a forward-pass smoke test after cross-version load is `pytest.skip()`'d unless a specific local OPFData cache path (`/data/OPF/opfdata_pyg_train`) exists — **this part of the fine-tuning path is not exercised in a fresh/CI environment**, only locally where that data happens to be present |

---

## 12. Root README vs. code: things worth flagging

- **"Registry" language**: nothing in the README literally claims a large
  model registry, but a reader could infer more variety than exists.
  There are exactly 2 checkpoints (§5); "Get the checkpoint" is a
  two-choice decision, not a catalog to browse.
- **hf_util and N-1 caching are conflated in the README's one-line
  pointer** ("See model/README.md#install... and cache customization for
  large N-1 sweeps" appears right after the `hf_util` section) — but the
  caches that matter for N-1 sweeps live in `cycle_basis.py`/
  `pe_features.py`/`dc_prior.py`, not in `hf_util/`. `hf_util` only
  downloads grid model / OPF-result JSON; it has no cache-sizing knobs of
  its own. See §6 for the precise story, including the more important
  point that this repo's own pipeline doesn't generate N-1 scenarios.
- **`predict_to_viewer.py`** exists, works, and is the only code in the
  repo that connects model output to the viewer's rendering — not
  mentioned in any README (model or viewer). See §9 and §14.

---

## 13. `power_grid/US/topology_solver_pipeline/` — Julia pipeline

**What this directory actually is** (its own `PIPELINE_DETAILS.md` is
explicit about this and it's worth restating precisely): it implements
**only stage 2** of a conceptual 4-stage pipeline — turning a raw
PowerModels-format topology JSON into a *cold-strict-AC-OPF-solvable*
JSON. Stage 1 (building the raw topology from OpenStreetMap/utility data)
is an upstream package not in this repo. **Stage 3** (the scenario
generator, `gen_perturbed_data.jl` / `export_gridsfm_data.jl`) happens to
live in the same directory for convenience but is conceptually separate —
it consumes stage 2's output and produces `.pyg.json`. Stage 4 is
`gridsfm` itself (§1–§11) consuming those files.

The pipeline is self-contained: its own `Project.toml`/`Manifest.toml`
live in this directory, so every Julia invocation uses `--project=.` with
no parent-project dependency resolution.

### 13.1 Stage 2 — `solve_topo_json.jl` (raw → `.solvable.json`)

```bash
julia --project=. solve_topo_json.jl <input.json> <output.solvable.json>
```

Iterates relaxation levels in escalation order **L0 → AC1 → L1 → L2 → L3
→ L4 → L5** (`LEVEL_ORDER = [0, 6, 1, 2, 3, 4, 5]` in the source — note
this order does **not** match the order levels are listed in
`shared/relaxation_levels.json`, which lists L0..L5 then AC1 last; the
JSON's array order is not the escalation order, only `solve_topo_json.jl`'s
hardcoded `LEVEL_ORDER` is). At each level: calls into
`run_opf_relaxation.jl`'s single-level mode, writes the mutated data to a
tmp file, reloads that tmp file, **zeros every warm-start field**
(`vm=1.0, va=0.0, pg=0.0, qg=0.0` — a genuine flat start), and re-solves
strict AC-OPF via Ipopt (`max_iter=10000, tol=1e-6, acceptable_tol=1e-4`).
The first level whose cold-strict re-solve reports `LOCALLY_SOLVED` /
`OPTIMAL` / `ALMOST_LOCALLY_SOLVED` wins and is written to the output
path; all mutated electrical parameters (rate_a, br_x, vmin/vmax, pmin,
shunts) are baked into the output JSON, plus a `_relaxation` metadata
field recording which level was used — downstream tools don't need to
re-run any relaxation logic, they can load the `.solvable.json` directly
with plain `PowerModels.parse_file` + `solve_ac_opf`.

**Relaxation levels** (`shared/relaxation_levels.json`, loaded by
`run_opf_relaxation.jl` as the single source of truth — confirmed present
and actually `include`d, not just a documentation artifact):

| level | what changes |
|---|---|
| L0 | model as-is, no relaxation |
| AC1 | V ∈ `[0.90, 1.10]`, Q limits ×1.5 (AC-OPF-only relaxation; not applied to DC formulations) |
| L1 | branch angle limits widened to ±60° |
| L2 | + branch ratings ×1.2 |
| L3 | ratings ×1.5, angles ±90°, `pmin` ×0.5 |
| L4 | + load capped at 70% of nameplate, `pmin` = 0 |
| L5 | thermal limits removed entirely, V ∈ `[0.85, 1.15]`, Q ×2.0, load cap 70%, `pmin` = 0 |

`run_opf_relaxation.jl` (1726 lines) is broader than the README's
relaxation-only framing suggests: it's a general-purpose OPF driver
supporting **AC, DC, and SOC** formulations (`--ac`/`--dc`/`--soc` CLI
flags), generator de-commitment, impedance-consistency fixes, and bounded
DC-derived shunt-compensation injection (AC1's shunt injection is
DC-derived, per the module docstring). It's never invoked directly as a
CLI entry point in this pipeline's own workflow (`solve_topo_json.jl`
`include()`s it and calls its internal functions); its CLI surface exists
for standalone use outside this pipeline.

### 13.2 Stage 3 — scenario generation (`.solvable.json` → `.pyg.json`)

**`export_gridsfm_data.jl <input> <output.pyg.json>`** — solves one grid
(anything `PowerModels.parse_file` accepts: `.m`, `.json`, `.raw`, ...)
with strict AC-OPF and writes one `.pyg.json`. Its `build_gridsfm_data`
function is the single source of truth for the schema and is `include()`d
and reused by the bulk generator below — there is exactly one code path
that produces this schema, not two independently-maintained ones. The
schema documented in its header (§3 cross-references this) additionally
includes a `solution` block (bus θ/V, generator Pg/Qg, branch flows) and
a `duals` block (bus power-balance + voltage-bound multipliers, generator
P/Q-limit multipliers, branch thermal-limit multipliers) that the Python
model package's loaders don't consume but the viewer and
`predict_to_viewer.py`'s output format mirror.

**`gen_perturbed_data.jl <grid_list_file> [n_proc] [out_root]`** — bulk
generator. Reads a grid-list file (`<solvable.json_path> <n_per_mode>`
per line), spawns `n_proc` (default `min(Sys.CPU_THREADS, 120)`) worker
processes via `Distributed.pmap`, and for every listed grid produces one
`base_unperturbed.pyg.json` plus `n_per_mode` scenarios for **each** of 5
pure perturbation modes (never combined within one scenario, so per-mode
signal stays uncorrelated):

| mode | mechanism |
|---|---|
| `loads` | one system-wide scale factor `sf ~ U(0.8, 1.5)`, then independent per-load ±10% jitter on both Pd and Qd |
| `costs` | shuffle cost-coefficient rows among ~40% of active generators, only within same-`ncost` pools (preserves each generator's cost-curve degree) |
| `killgen` | flip `gen_status=0` on 1/2/3 generators with probability 0.7/0.2/0.1, but only among generators with `pmax > 0.01`, and never below 2 remaining active generators |
| `derate` | scale `rate_a`/`rate_b`/`rate_c` by one factor `~U(0.7, 0.95)` on ~10% of in-service branches |
| `vsqueeze` | on ~10% of buses, shrink `vmin`/`vmax` independently by `~U(0, 0.01)` pu on each boundary (reverted if the shrink would cross `vmin ≥ vmax`) |

Task queue is **global across all listed grids** (workers never idle
waiting on one case to finish) and **resume-friendly** — existing output
files are skipped, so a killed/restarted run doesn't redo completed work.
Total scenarios per grid = `1 + 5·n_per_mode`. Per-scenario RNG seed is
derived from `42 + scenario_idx + hash((file, mode), UInt(0))` — the
explicit `UInt(0)` seed to Julia's `hash` is because `hash`'s default salt
is randomized per Julia session, so omitting it would make scenario
generation non-reproducible run-to-run; this fixed-salt call makes it
reproducible.

**`run_gen_gridsfm_data.sh [n_proc=51] [out_root=./out]
[grid_list=$SCRIPT_DIR/grids_solvable.txt] [cpu_range=77-127]`** — thin
positional-arg wrapper around `gen_perturbed_data.jl`; the `cpu_range`
default (`77-127`, i.e. 51 cores) is tuned for a specific large multi-socket
build host, not a general default — override it for any other machine.

### 13.3 Verification

**`solve_pyg_json.jl <solvable.json> <scenario.pyg.json>`** — the
integrity check for stage-3 output: reconstructs the exact PowerModels
data the scenario represents (loads/killgen/derate/vsqueeze/costs values
overlaid from the specific `.pyg.json` columns listed in
`PIPELINE_DETAILS.md`, warm-started from the scenario's own recorded
solution), re-solves strict AC-OPF, and checks the objective matches
`metadata.objective` within 0.1%. Exit code 0 = pass. This is what "does
this `.pyg.json` contain enough information to exactly reproduce the
solve that produced it" actually means operationally.

**`integration_test_all_components.sh <raw_input.json>
[out_dir=/tmp/topo_solver_pipe_test]`** — the end-to-end CI gate: solve →
export → re-solve-and-compare (< 0.01%) → perturb (`n_per_mode=1`) →
`solve_pyg_json.jl` on all 6 outputs (< 0.1% each), plus an optional
Python sanity check (`build_hetero_data_from_json`, skipped gracefully if
`gridfm`/`python3` isn't importable on the host).

### 13.4 Preprocessing step not listed among the "4 stages"

`docker/patch_model.jl <input.json> [output.json]` — injects empty
`"storage"`/`"switch"` dicts into a model JSON that predates a
PowerModels.jl v0.21+ requirement for those keys to be present (even if
empty; if `output.json` is omitted the input is patched in place). This is
**not** one of the four stages described in `PIPELINE_DETAILS.md` (it's
absent from that document's stage list entirely), meaning models from the
`GridSFM_US_power_grid` data release, as currently exported, are **not
directly loadable by the pinned PowerModels.jl version** without this
patch.

Despite not being a documented "stage", the Makefile (verified directly,
not from the README) wires it in everywhere it's actually needed: `run` /
`local-run` (via `docker/run_pipeline.sh`'s `[PREPROCESS]` step), the
standalone `patch` / `local-patch` targets, and — contrary to what an
inference from the stage table alone might suggest — **`solve` and
`local-solve` also run `patch_model.jl` first**, chained via `&&` before
`solve_topo_json.jl` in the same Makefile recipe, as do `integration-test`
/ `local-integration-test`. The only targets that legitimately skip it are
`export`, `perturb`, `verify(-one)`, and `gen-bulk` / `local-gen-bulk`,
because those operate on an already-`.solvable.json` (hence
already-patched) file, not the raw data-release model. So despite the
awkward "stage 1.5, undocumented in `PIPELINE_DETAILS.md`" positioning,
this is **not** currently a footgun in the Makefile-driven workflow — every
Makefile target that consumes a raw `*_model.json` patches it first. The
`README.md`'s own stage table (§3, "Run Individual Stages") lists "Patch
model for compatibility" as its own row (`make patch` / `make local-patch`)
ahead of "Solve", which is consistent with this. The residual risk is
narrower than "don't use `make solve`": it's specifically for anyone
invoking `solve_topo_json.jl` **directly** with `julia`, bypassing the
Makefile entirely, against an unpatched data-release file.

### 13.5 Makefile / Docker orchestration

Every stage has a Docker target and a `local-`-prefixed native-Julia
counterpart (`make solve` / `make local-solve`, etc. — full mapping table
in the pipeline README). Key config variables (all overridable on the
command line): `DATA_DIR` (must contain `04h/`/`16h/` subdirs of
`*_model.json`), `OUTPUT_DIR` (default `./output`), `STATE` (default
`alabama`), `HOUR` (default `16h`), `N_PER_MODE` (default `1`),
`N_WORKERS` (default `2`), `CPU_RANGE` (auto-detected `0-(N-1)`, Linux-only
`taskset` pinning, no-op on macOS), `GRID_LIST` (required for
`gen-bulk`/`local-gen-bulk`). Docker mounts `DATA_DIR` read-only and
`OUTPUT_DIR` read-write; a documented footgun is that Docker writes
output as root, so `make clean`/`clean-state` themselves shell back into
Docker to remove root-owned files rather than trying to `rm -rf` as the
host user.

The Docker image (`docker/Dockerfile`) is `julia:1.11-bookworm` +
`python3`/`pip` (for the integration test's optional Python check) +
the pinned `Manifest.toml` packages (`Pkg.instantiate(); Pkg.precompile()`
at build time — the slow, cached layer) + a package-load smoke check at
build time (`using PowerModels, Ipopt, JuMP, JSON`).

---

## 14. `power_grid/US/viewer/` — data viewer

Pure Python stdlib HTTP server (`serve.py`, `http.server` +
`ThreadingMixIn`) + a single static `index.html` (731 lines, vanilla JS,
Leaflet.js from CDN — **no** build step, no npm, no bundler). Requires
Python 3.10+, no pip packages.

```bash
cd power_grid/US/viewer
python serve.py --data-dir /path/to/gridsfm_data --port 8050   # defaults: ../, 8050
```

`--data-dir` must contain `16h/`/`04h/` subdirectories with
`<state>_model.json` (+ optionally `<state>_dc_results.json`,
`<state>_ac_results.json`, `<state>_interfaces.json`) — exactly what
`GridSFM_PG_Loader.download_all()` / `export_dir` produces (§7), or what
the topology pipeline's `export_gridsfm_data.jl` output can be renamed
into. `serve.py` refuses to start if no hour-pattern directory is found
under `--data-dir`, printing the expected layout. Serving is path-
sandboxed (`file_path.is_relative_to(data_dir)` check before reading —
prevents `../`-style traversal via the `/data/<hour>/<filename>` route)
and only `.json` files are served through that route.

Two small discovery APIs the front end depends on: `GET /api/states`
(scans both hour dirs for `*_model.json`, derives the "is this a
multi-state region vs a single state" flag from a **hardcoded** set of
region-name strings — `new_england, pacific_nw, desert_sw, western,
eastern, pjm, ercot, miso, spp, continental_us` — not derived from any
metadata field) and `GET /api/hours` (any `\d+h$`-named subdirectory,
converted to a 12-hour AM/PM label).

**Four views** (README-documented): Network Model (topology on a Leaflet
map, color-coded by voltage level via a `kvColor` bucket function),
OPF Summary (generation mix / cost / solver status), Economic Dispatch
(per-generator dispatch stacked by fuel type, `fuelColor`/`FUEL_ALIASES`
mapping), Line Congestion (branch loading vs. thermal rating).

**Undocumented capability — a three-way DC/AC/GSFM results toggle,
absent from the viewer README entirely**: the Economic Dispatch and Line
Congestion views both fetch a third results file,
`<state>_gridsfm_results.json` (`resultsDir() + currentState +
'_gridsfm_results.json'`), alongside the DC and AC results files, and
render a toggle button row (`opfToggleHTML`) with three buttons labeled
DC / AC / **GSFM** (`title="GridSFM ML surrogate"`), each disabled when
its corresponding data file isn't present for the selected state/hour.
This is exactly the file that `model/examples/predict_to_viewer.py`
(§9) produces — so the intended workflow is: run the topology pipeline
to get a `.pyg.json`, run `predict_to_viewer.py` against it to get
`<state>_gridsfm_results.json`, drop that file next to the DC/AC results
in the viewer's data directory, and the viewer will let a user flip
between solver-computed and ML-predicted dispatch/congestion for the
same grid. Neither the viewer README nor the top-level README mentions
this integration exists.

---

## 15. Known limitations

- **Grid size floor**: GridSFM-Open is trained on grids `>=500` buses;
  smaller cases (`case14_ieee`, `case30_ieee`, `case57_ieee`,
  `case118_ieee`) are explicitly out of distribution and both example
  scripts warn about this — the model will run on them without error but
  the output is not claimed to be meaningful.
- **Fine-tuning is narrowly scoped**: v1.1 checkpoint only; OPFData
  dataset format only. There is no supported path to fine-tune on
  `.pyg.json` scenarios produced by this repo's own
  `topology_solver_pipeline`, despite that pipeline being the repo's own
  data-generation tool — the loss/dataset code's column conventions are
  pinned to OPFData's schema, not GridSFM's own `.pyg.json` schema. A
  cross-version load of v1.0 for fine-tuning starts with zeroed max-pool
  fusion channels and produces a checkpoint that is no longer v1.0-shaped.
- **No built-in N-1/contingency scenario generation.** Despite the model
  package's caching guidance explicitly targeting "large N-1 / N-k
  sweeps" (§6), nothing in `topology_solver_pipeline` generates
  line-outage contingencies; its five perturbation modes cover load,
  cost, generator-outage, derating, and voltage-band changes only. N-1
  topology data in this repo comes solely from OPFData's external `n1`
  split (fine-tuning/eval only, not inference-input generation).
  Constructing an N-1 sweep of `.pyg.json` inputs is left entirely to
  the user.
- **Branch flows are physics-derived, not learned outputs** — if the
  predicted θ/V are inaccurate, the derived flows inherit that error
  deterministically; there is no independent flow-correction head in the
  inference path (the loss does train the θ/V heads against flow
  consistency during fine-tuning, but at inference time flows are pure
  post-processing).
- **Single-GPU only for fine-tuning** — `compute_loss`/`finetune_opfdata`
  have no distributed-training plumbing (`loss.py`'s module docstring
  states this directly: "Single-GPU only (no DDP `find_unused_parameters`
  plumbing)").
- **`OPFDataAdapterDataset(n_graphs=...)` doesn't shortcut the expensive
  part** — capping the graph count still pays the full first-time
  download/decode cost of the underlying `OPFDataset` shard; there's no
  lightweight "just get me 16 graphs" path.
- **Viewer has no data-generation or write path** — it's read-only,
  serving whatever `.json` files exist in the data directory; it cannot
  trigger the Julia pipeline or the model itself. The DC/AC/GSFM toggle
  only works if those specific result files were placed there out of
  band.
- **`patch_model.jl` is a prerequisite for data-release models that is
  undocumented in `PIPELINE_DETAILS.md`'s stage list, but is correctly
  wired into every relevant Makefile target** (§13.4, verified against the
  current `Makefile`) — `solve`/`local-solve`, `run`/`local-run`, and
  `integration-test`/`local-integration-test` all run it automatically
  before solving. The only real gap is for a user who invokes
  `solve_topo_json.jl` directly with `julia`, bypassing the Makefile
  entirely, against an unpatched `GridSFM_US_power_grid` release file.
- **This document's Julia-pipeline coverage is from reading source, not
  running it** — the pipeline requires Julia 1.11+, Ipopt, and (for
  Docker) building a multi-hundred-MB image; none of that was executed
  as part of producing this reference. Stage behavior, CLI flags, and
  output schemas are transcribed faithfully from the scripts' own code
  and comments, but end-to-end runtime behavior (solver convergence
  characteristics, actual wall-clock timings beyond the README's own
  "10+ minutes for Texas/California" note) was not independently verified.
- **`relaxation_levels.json`'s array order does not reflect the actual
  escalation order** used by `solve_topo_json.jl` (L0→AC1→L1→...→L5); a
  reader parsing the JSON file's `levels` array top-to-bottom would infer
  the wrong escalation sequence (it lists AC1 last).
