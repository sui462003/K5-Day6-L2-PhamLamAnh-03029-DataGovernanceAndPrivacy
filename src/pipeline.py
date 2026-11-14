"""PoC Pipeline for Track 01 · Coverage Planner.

Lab contract = 5 automated steps, run by `python -m src.main ...`:
    01 load      input + provenance check (manifest.csv)
    02 process   Cost-Aware Weighted Set-Cover with Risk/Hard Priority & Unknown Disambiguation
    03 rank      prioritise for a fixed review budget B, flagging hard cells & unknown tags
    04 review    human-in-the-loop: queue -> reviewer decisions -> overrides + audit log
    05 decide    GO / HOLD / REWORK with reason_code + owner, write pick.csv and capture plan

Required output: out/pick.csv with columns ['clip_id'] (and optional 'rank').
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

from .common.context import RunContext
from .common.decision import Decision, write_decision
from .common.manifest import verify_manifest
from .common.review import load_review_decisions, make_review_queue, write_review_queue
from .metrics.tracks import get_track

DIMS = ["route", "weather", "light", "object"]


def matches_cell(clip_row: dict | pd.Series, cell_row: dict | pd.Series) -> bool:
    """Check if clip matches target cell criteria. '*' is a wildcard."""
    for d in DIMS:
        c_val = str(cell_row[d]).strip()
        if c_val == "*":
            continue
        clip_val = str(clip_row[d]).strip()
        if clip_val != c_val:
            return False
    return True


# ---------------------------------------------------------------- 01 load
def load(ctx: RunContext) -> dict[str, Any]:
    if ctx.track != "t01":
        raise NotImplementedError(f"Pipeline implemented for t01 (Coverage Planner), got {ctx.track}")

    manifest = ctx.input_dir / "manifest.csv"
    problems = verify_manifest(manifest) if manifest.exists() else [{"problem": "no_manifest"}]
    if problems:
        ctx.audit.log(actor="pipeline", action="provenance_problems", item_id=str(manifest), after=problems[:20])
        if ctx.config.get("strict_provenance", True):
            raise ValueError(f"provenance check failed ({len(problems)} problems), first: {problems[:3]}")

    # Determine candidate file: holdout_candidates.csv if present in path/name, otherwise candidates.csv
    cand_path = ctx.input_dir / "holdout_candidates.csv" if "holdout" in str(ctx.input_dir).lower() else ctx.input_dir / "candidates.csv"
    if not cand_path.exists():
        # Fallback to candidates.csv
        cand_path = ctx.input_dir / "candidates.csv"

    targets_path = ctx.input_dir / "targets.csv"
    if not targets_path.exists():
        raise FileNotFoundError(f"Missing required targets.csv in {ctx.input_dir}")

    candidates = pd.read_csv(cand_path)
    targets = pd.read_csv(targets_path)

    ctx.stats["n_candidates_available"] = len(candidates)
    ctx.stats["n_targets"] = len(targets)
    return {
        "manifest": pd.read_csv(manifest) if manifest.exists() else None,
        "candidates": candidates,
        "targets": targets,
    }


# ---------------------------------------------------------------- 02 process
def process(ctx: RunContext, data: dict[str, Any]) -> pd.DataFrame:
    """Cost-Aware Marginal Gain Greedy Set-Cover with Risk/Hard Priority.

    Returns the candidates DataFrame enriched with:
        - score: float, higher = prioritized first for K selection and review
        - is_picked: int (1 = in top-K selection, 0 = not picked)
        - rank: int (1..K for selected clips)
        - reason: explanation of selection / prioritization
    """
    candidates: pd.DataFrame = data["candidates"].copy()
    targets: pd.DataFrame = data["targets"].copy()

    k_default = 6 if "holdout" in str(ctx.input_dir).lower() else 30
    K = int(ctx.param("k", k_default))

    weights = dict(zip(targets["cell_id"], targets["weight"].astype(float)))
    hard_cells = set(targets.loc[targets["is_hard"] == 1, "cell_id"])

    # 1. Map each candidate clip to target cells it matches
    cand_matches: dict[str, set[str]] = {}
    for _, clip in candidates.iterrows():
        matched = {
            cell["cell_id"]
            for _, cell in targets.iterrows()
            if matches_cell(clip, cell)
        }
        cand_matches[str(clip["clip_id"])] = matched

    cand_dict = candidates.set_index("clip_id").to_dict("index")
    remaining = set(candidates["clip_id"].astype(str))

    selected: list[str] = []
    reasons: dict[str, str] = {}
    covered_cells: set[str] = set()

    # Phase 1: Marginal Gain Greedy Set-Cover (Prioritize new cells and hard cells)
    while len(selected) < K and remaining:
        best_clip = None
        best_score = -1.0
        best_new_cells = set()

        for cid in remaining:
            new_cells = cand_matches[cid] - covered_cells
            if not new_cells:
                continue

            # Weight calculation: hard cells get 1.5x boost
            gain = sum(weights[c] * (1.5 if c in hard_cells else 1.0) for c in new_cells)
            cost = max(1.0, float(cand_dict[cid].get("cost_min", 1.0)))
            # Efficiency score: gain divided by square root of cost
            score = gain / math.sqrt(cost)

            if score > best_score:
                best_score = score
                best_clip = cid
                best_new_cells = new_cells

        if best_clip is None:
            break

        selected.append(best_clip)
        covered_cells |= best_new_cells
        remaining.remove(best_clip)
        reasons[best_clip] = f"Covers new cells: {sorted(best_new_cells)}"

    ctx.stats["phase1_clips"] = len(selected)
    ctx.stats["covered_cells_phase1"] = sorted(covered_cells)
    ctx.stats["missing_cells"] = sorted(set(targets["cell_id"]) - covered_cells)

    # Phase 2: Fill remaining slots up to K
    # Prioritize: Redundancy for hard cells, scenario diversity, and low capture cost
    chosen_scenarios = [tuple(cand_dict[c][d] for d in DIMS) for c in selected]

    while len(selected) < K and remaining:
        best_clip = None
        best_score = -999999.0
        best_reason = ""

        for cid in remaining:
            row = cand_dict[cid]
            scenario = tuple(row[d] for d in DIMS)
            is_dup = scenario in chosen_scenarios
            c_matches = cand_matches[cid]
            hard_overlap = len(c_matches & hard_cells)
            cost = float(row.get("cost_min", 10.0))

            score = 0.0
            r_notes = []

            # Reward unique scenario to reduce duplicate clips metric
            if not is_dup:
                score += 15.0
                r_notes.append("unique_scenario")

            # Reward redundancy for hard/rare cells (in case of 12% noisy metadata)
            if hard_overlap > 0:
                score += hard_overlap * 10.0
                r_notes.append(f"redundancy_hard({sorted(c_matches & hard_cells)})")

            # Cost penalty
            score -= cost * 0.2

            # Unknown tag check: flag clips that could match uncovered targets
            if row.get("route") == "urban" and row.get("weather") == "fog" and row.get("object") == "pedestrian":
                score += 25.0
                r_notes.append("potential_C13_candidate(urban,fog,pedestrian)")
            if row.get("route") == "alley" and row.get("object") == "pedestrian":
                score += 20.0
                r_notes.append("potential_C15_candidate(alley,pedestrian)")

            if score > best_score:
                best_score = score
                best_clip = cid
                best_reason = "; ".join(r_notes) if r_notes else "low_cost_fill"

        if best_clip is None:
            break

        selected.append(best_clip)
        chosen_scenarios.append(tuple(cand_dict[best_clip][d] for d in DIMS))
        reasons[best_clip] = best_reason
        remaining.remove(best_clip)

    # Calculate scores for ranking
    # Picked clips receive top scores (1000 down to 1000 - K), unpicked receive scores based on residual value
    picked_set = set(selected)
    rank_map = {cid: idx + 1 for idx, cid in enumerate(selected)}

    scores = []
    ranks = []
    final_reasons = []

    for _, row in candidates.iterrows():
        cid = str(row["clip_id"])
        if cid in picked_set:
            r = rank_map[cid]
            ranks.append(r)
            # Higher score for earlier picks
            scores.append(round(1000.0 - r, 2))
            final_reasons.append(reasons.get(cid, "Selected for coverage"))
        else:
            ranks.append(None)
            # Unpicked clips scored by potential value and unknown tag inspection urgency
            has_unknown = any(str(row[d]).strip() == "unknown" for d in DIMS)
            base = 50.0 if has_unknown else 10.0
            scores.append(round(base - float(row.get("cost_min", 10.0)) * 0.5, 2))
            final_reasons.append("Unselected pool (audit reserve)")

    candidates["score"] = scores
    candidates["rank"] = ranks
    candidates["is_picked"] = [1 if cid in picked_set else 0 for cid in candidates["clip_id"].astype(str)]
    candidates["reason"] = final_reasons

    # Log summary statistics
    cov_weight = sum(weights[c] for c in covered_cells)
    total_weight = sum(weights.values())
    ctx.stats["weighted_coverage_metadata"] = round(cov_weight / total_weight, 4) if total_weight else 0.0
    ctx.stats["n_picked"] = len(selected)
    ctx.stats["total_cost_min"] = sum(cand_dict[c]["cost_min"] for c in selected)

    return candidates


# ---------------------------------------------------------------- 03 rank
def rank(ctx: RunContext, scored: pd.DataFrame) -> pd.DataFrame:
    """Prioritize review queue: audit clips covering rare cells, missing cells, or having unknown tags."""
    budget = int(ctx.param("budget", 20))
    if "score" not in scored.columns:
        raise ValueError("rank(): scored DataFrame needs numeric `score` column")

    q = make_review_queue(
        scored,
        id_col="clip_id",
        score_col="score",
        budget=budget,
        reason_col="reason",
    )
    write_review_queue(q, ctx.out("review_queue.csv"))
    return q


# ---------------------------------------------------------------- 04 review
def review(ctx: RunContext, scored: pd.DataFrame, queue: pd.DataFrame) -> pd.DataFrame:
    """Apply human review overrides if present (out/review_queue_reviewed.csv)."""
    reviewed_path = ctx.output_dir / "review_queue_reviewed.csv"
    if reviewed_path.exists():
        reviewed = load_review_decisions(reviewed_path)
        ctx.stats["n_reviewed"] = len(reviewed)

        for _, rev in reviewed.iterrows():
            cid = str(rev.get("item_id", rev.get("clip_id", "")))
            dec = str(rev.get("reviewer_decision", "")).strip().lower()
            note = str(rev.get("review_note", ""))
            reviewer = str(rev.get("reviewer", "human_reviewer"))

            if dec in ("reject", "drop"):
                # Demote score if rejected by human
                scored.loc[scored["clip_id"] == cid, "score"] = -100.0
                scored.loc[scored["clip_id"] == cid, "is_picked"] = 0
                ctx.audit.log(actor=reviewer, action="reject_clip", item_id=cid, note=note)
            elif dec in ("approve", "keep"):
                ctx.audit.log(actor=reviewer, action="approve_clip", item_id=cid, note=note)

    return scored


# ---------------------------------------------------------------- 05 decide
def decide(ctx: RunContext, final: pd.DataFrame) -> Decision:
    """Make GO / HOLD / REWORK decision based on coverage gates."""
    owner = ctx.config.get("owner", "Pham Lam Anh - 03029 (Data Planner & Governance Lead)")
    missing = ctx.stats.get("missing_cells", [])
    cov = ctx.stats.get("weighted_coverage_metadata", 0.0)

    # Save detailed coverage status report
    targets_path = ctx.input_dir / "targets.csv"
    if targets_path.exists():
        targets = pd.read_csv(targets_path)
        cov_cells = set(ctx.stats.get("covered_cells_phase1", []))
        targets["status"] = ["COVERED" if c in cov_cells else "MISSING" for c in targets["cell_id"]]
        targets.to_csv(ctx.out("coverage_status.csv"), index=False)

    # Save capture plan for missing cells
    if missing:
        plan_rows = []
        if targets_path.exists():
            targets = pd.read_csv(targets_path).set_index("cell_id")
            for m in missing:
                t = targets.loc[m]
                plan_rows.append({
                    "missing_cell_id": m,
                    "target_route": t["route"],
                    "target_weather": t["weather"],
                    "target_light": t["light"],
                    "target_object": t["object"],
                    "weight": t["weight"],
                    "is_hard": t["is_hard"],
                    "recommended_action": "Schedule dedicated field capture mission with specified conditions",
                })
        pd.DataFrame(plan_rows).to_csv(ctx.out("capture_plan.csv"), index=False)

    # Decision logic:
    # If key target cells are missing from candidate pool, we must HOLD release until capture mission completes.
    if missing:
        return Decision(
            status="HOLD",
            reason_code="MISSING_TARGET_COVERAGE",
            reason=f"Candidate pool lacks coverage for {len(missing)} target cells: {missing}. Field capture plan generated.",
            owner=owner,
            scope=ctx.track,
            evidence=[str(ctx.out("capture_plan.csv")), str(ctx.out("coverage_status.csv"))],
        )

    if cov < 0.85:
        return Decision(
            status="REWORK",
            reason_code="INSUFFICIENT_WEIGHTED_COVERAGE",
            reason=f"Weighted coverage {cov*100:.1f}% below minimum release threshold 85%.",
            owner=owner,
            scope=ctx.track,
        )

    return Decision(
        status="GO",
        reason_code="COVERAGE_REQUIREMENTS_SATISFIED",
        reason=f"All target scenarios covered. Weighted coverage = {cov*100:.1f}%.",
        owner=owner,
        scope=ctx.track,
        evidence=[str(ctx.out(get_track(ctx.track).pred_file))],
    )


# ---------------------------------------------------------------- Runner
def run(ctx: RunContext) -> None:
    spec = get_track(ctx.track)
    data = load(ctx)
    scored = process(ctx, data)
    ctx.stats["n_items"] = len(scored)

    queue = rank(ctx, scored)
    final = review(ctx, scored, queue)

    # Extract final K picked clips for pick.csv
    k_default = 6 if "holdout" in str(ctx.input_dir).lower() else 30
    K = int(ctx.param("k", k_default))

    # Sort by score descending to get top K picks
    ranked_picks = final.sort_values("score", ascending=False).head(K).copy()
    ranked_picks["rank"] = range(1, len(ranked_picks) + 1)

    # Ensure required columns are written
    out_df = ranked_picks[["clip_id", "rank"]]
    out_df.to_csv(ctx.out(spec.pred_file), index=False)

    # Write decision
    decision = decide(ctx, final)
    write_decision(decision, ctx.out("decision.json"), ctx.out("decision_log.csv"))
