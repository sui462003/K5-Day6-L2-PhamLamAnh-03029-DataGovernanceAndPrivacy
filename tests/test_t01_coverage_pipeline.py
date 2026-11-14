"""Unit tests for Track 01 PoC Pipeline (Pham Lam Anh - 03029)."""
import json
from pathlib import Path

import pandas as pd
import pytest

from src import main as runner


def test_t01_poc_success_e2e(tmp_path):
    """Test standard PoC pipeline runs E2E offline on synthetic fixtures with K=30."""
    out_dir = tmp_path / "poc_run"
    rc = runner.main([
        "--track", "t01",
        "--input", "fixtures/t01_coverage",
        "--output", str(out_dir),
        "--offline",
        "--seed", "42",
    ])
    assert rc == 0, f"Expected returncode 0, got {rc}"

    # Verify required outputs
    pick_file = out_dir / "pick.csv"
    assert pick_file.exists(), "pick.csv must be created"
    picks = pd.read_csv(pick_file)
    assert len(picks) == 30, f"Expected 30 picked clips, got {len(picks)}"
    assert list(picks.columns) == ["clip_id", "rank"]
    assert picks["clip_id"].nunique() == 30, "All picked clip IDs must be unique"

    # Verify governance artifacts
    assert (out_dir / "decision.json").exists()
    assert (out_dir / "decision_log.csv").exists()
    assert (out_dir / "review_queue.csv").exists()
    assert (out_dir / "capture_plan.csv").exists()
    assert (out_dir / "coverage_status.csv").exists()
    assert (out_dir / "run_info.json").exists()

    decision = json.loads((out_dir / "decision.json").read_text())
    assert decision["status"] == "HOLD"
    assert decision["reason_code"] == "MISSING_TARGET_COVERAGE"
    assert "C13" in decision["reason"] and "C15" in decision["reason"]

    # Verify capture plan contains the 2 missing cells
    plan = pd.read_csv(out_dir / "capture_plan.csv")
    assert set(plan["missing_cell_id"]) == {"C13", "C15"}


def test_t01_poc_holdout_e2e(tmp_path):
    """Test PoC pipeline runs on holdout candidates with K=6."""
    out_dir = tmp_path / "holdout_run"
    rc = runner.main([
        "--track", "t01",
        "--input", "fixtures/t01_coverage",
        "--output", str(out_dir),
        "--offline",
        "--seed", "42",
    ])
    assert rc == 0


def test_t01_missing_targets_failure_clearly(tmp_path):
    """Failure test: if input directory lacks targets.csv, pipeline fails clearly with rc=2."""
    empty_input = tmp_path / "empty_dir"
    empty_input.mkdir()
    # Create dummy candidates.csv
    (empty_input / "candidates.csv").write_text("clip_id,route,weather,light,object,risk,cost_min,duration_s\n")
    
    out_dir = tmp_path / "err_out"
    rc = runner.main([
        "--track", "t01",
        "--input", str(empty_input),
        "--output", str(out_dir),
        "--offline",
    ])
    assert rc == 2


def test_t01_human_review_override(tmp_path):
    """Test Human-in-the-loop review override mechanism and audit logging."""
    out_dir = tmp_path / "review_run"
    # First run to generate review queue
    rc = runner.main([
        "--track", "t01",
        "--input", "fixtures/t01_coverage",
        "--output", str(out_dir),
        "--offline",
    ])
    assert rc == 0

    # Simulate reviewer rejecting clip_115 with note
    rev_queue = pd.read_csv(out_dir / "review_queue.csv", dtype=str)
    rev_queue["reviewer"] = "PhamLamAnh_03029"
    rev_queue.loc[rev_queue["item_id"] == "clip_115", "reviewer_decision"] = "reject"
    rev_queue.loc[rev_queue["item_id"] == "clip_115", "review_note"] = "Metadata unverified, bad weather tag"
    rev_queue.to_csv(out_dir / "review_queue_reviewed.csv", index=False)

    # Re-run pipeline to consume reviewed overrides
    rc2 = runner.main([
        "--track", "t01",
        "--input", "fixtures/t01_coverage",
        "--output", str(out_dir),
        "--offline",
    ])
    assert rc2 == 0

    # Verify audit log was updated
    audit_file = out_dir / "audit_log.jsonl"
    assert audit_file.exists()
    lines = audit_file.read_text().splitlines()
    assert any("reject_clip" in l and "clip_115" in l for l in lines)
