from pathlib import Path

code = r'''from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

VERSION = "v1.0.1"
DEFAULT_PYTHON = sys.executable


@dataclass
class CampaignShardRecord:
    shard_index: int
    start: int
    end: int
    outdir: str
    status: str
    started_at_utc: str
    finished_at_utc: Optional[str]
    elapsed_sec: Optional[float]
    exit_code: Optional[int]
    summary_path: Optional[str]
    processed_jobs: Optional[int]
    results_count: Optional[int]
    no_trade_jobs: Optional[int]
    error_jobs: Optional[int]
    total_elapsed_sec: Optional[float]
    note: Optional[str]


@dataclass
class CampaignState:
    version: str
    manifest: str
    runner: str
    outroot: str
    shard_size: int
    total_jobs: int
    next_start: int
    finished_shards: int
    last_status: str
    updated_at_utc: str
    last_outdir: Optional[str]
    last_summary_path: Optional[str]


class CampaignError(RuntimeError):
    pass


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class JsonlManifestIndex:
    def __init__(self, manifest_path: Path) -> None:
        self.manifest_path = manifest_path

    def count_jobs(self) -> int:
        count = 0
        with self.manifest_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    count += 1
        return count


class CampaignOrchestrator:
    def __init__(self, args: argparse.Namespace) -> None:
        self.manifest = Path(args.manifest)
        self.runner = Path(args.runner)
        self.outroot = Path(args.outroot)
        self.shard_size = int(args.shard_size)
        self.python_exe = args.python_exe
        self.max_shards = args.max_shards
        self.sleep_sec = float(args.sleep_sec)
        self.stop_on_error = bool(args.stop_on_error)
        self.force_restart_current = bool(args.force_restart_current)
        self.dry_run = bool(args.dry_run)
        self.start_override = args.start
        self.end_override = args.end

        self.campaign_state_path = self.outroot / "campaign_state.json"
        self.campaign_history_path = self.outroot / "campaign_history.jsonl"
        self.campaign_summary_path = self.outroot / "campaign_summary.json"

        self._validate_inputs()
        self.outroot.mkdir(parents=True, exist_ok=True)
        self.total_jobs = JsonlManifestIndex(self.manifest).count_jobs()
        self.state = self._load_or_init_state()

    def _validate_inputs(self) -> None:
        if not self.manifest.exists():
            raise CampaignError(f"manifest_not_found: {self.manifest}")
        if not self.runner.exists():
            raise CampaignError(f"runner_not_found: {self.runner}")
        if self.shard_size <= 0:
            raise CampaignError("shard_size_must_be_positive")
        if self.start_override is not None and self.start_override < 0:
            raise CampaignError("start_must_be_non_negative")
        if self.end_override is not None and self.end_override <= 0:
            raise CampaignError("end_must_be_positive")
        if self.start_override is not None and self.end_override is not None and self.start_override >= self.end_override:
            raise CampaignError("start_must_be_less_than_end")
        if self.max_shards is not None and self.max_shards <= 0:
            raise CampaignError("max_shards_must_be_positive")

    def _load_or_init_state(self) -> CampaignState:
        if self.campaign_state_path.exists():
            payload = json.loads(self.campaign_state_path.read_text(encoding="utf-8"))
            state = CampaignState(**payload)
            if state.manifest != str(self.manifest):
                raise CampaignError(
                    f"campaign_state_manifest_mismatch: state={state.manifest} current={self.manifest}"
                )
            if state.runner != str(self.runner):
                raise CampaignError(
                    f"campaign_state_runner_mismatch: state={state.runner} current={self.runner}"
                )
            state.total_jobs = self.total_jobs
            state.updated_at_utc = utc_now_iso()
            return state

        next_start = self.start_override or 0
        return CampaignState(
            version=VERSION,
            manifest=str(self.manifest),
            runner=str(self.runner),
            outroot=str(self.outroot),
            shard_size=self.shard_size,
            total_jobs=self.total_jobs,
            next_start=next_start,
            finished_shards=0,
            last_status="initialized",
            updated_at_utc=utc_now_iso(),
            last_outdir=None,
            last_summary_path=None,
        )

    def _save_state(self) -> None:
        self.state.updated_at_utc = utc_now_iso()
        self.campaign_state_path.write_text(
            json.dumps(asdict(self.state), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _append_history(self, record: CampaignShardRecord) -> None:
        with self.campaign_history_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    def _write_campaign_summary(self) -> None:
        history = self._load_history()
        finished = [x for x in history if x.get("status") == "done"]
        errored = [x for x in history if x.get("status") == "error"]
        skipped = [x for x in history if x.get("status") == "skipped_existing"]

        payload = {
            "version": VERSION,
            "generated_at_utc": utc_now_iso(),
            "manifest": str(self.manifest),
            "runner": str(self.runner),
            "outroot": str(self.outroot),
            "shard_size": self.shard_size,
            "total_jobs": self.total_jobs,
            "next_start": self.state.next_start,
            "history_rows": len(history),
            "finished_shards": len(finished),
            "error_shards": len(errored),
            "skipped_existing_shards": len(skipped),
            "sum_processed_jobs": sum(int(x.get("processed_jobs") or 0) for x in history),
            "sum_results_count": sum(int(x.get("results_count") or 0) for x in history),
            "sum_no_trade_jobs": sum(int(x.get("no_trade_jobs") or 0) for x in history),
            "sum_error_jobs": sum(int(x.get("error_jobs") or 0) for x in history),
            "sum_total_elapsed_sec": round(sum(float(x.get("elapsed_sec") or 0.0) for x in history), 3),
            "last_status": self.state.last_status,
            "last_outdir": self.state.last_outdir,
            "last_summary_path": self.state.last_summary_path,
        }
        self.campaign_summary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _load_history(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        if not self.campaign_history_path.exists():
            return rows
        with self.campaign_history_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    def _shard_outdir(self, start: int, end: int) -> Path:
        return self.outroot / f"shard_{start:06d}_{end:06d}"

    def _read_summary(self, summary_path: Path) -> Dict[str, Any]:
        if not summary_path.exists():
            return {}
        try:
            return json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _infer_existing_done(self, outdir: Path) -> Optional[CampaignShardRecord]:
        summary_path = outdir / "summary.json"
        summary = self._read_summary(summary_path)
        if not summary:
            return None

        processed_jobs = int(summary.get("processed_jobs") or 0)
        results_count = int(summary.get("results_count") or 0)
        no_trade_jobs = int(summary.get("no_trade_jobs") or 0)
        error_jobs = int(summary.get("error_jobs") or 0)
        total_elapsed_sec = float(summary.get("total_elapsed_sec") or 0.0)

        return CampaignShardRecord(
            shard_index=-1,
            start=-1,
            end=-1,
            outdir=str(outdir),
            status="skipped_existing",
            started_at_utc=utc_now_iso(),
            finished_at_utc=utc_now_iso(),
            elapsed_sec=0.0,
            exit_code=0,
            summary_path=str(summary_path) if summary_path.exists() else None,
            processed_jobs=processed_jobs,
            results_count=results_count,
            no_trade_jobs=no_trade_jobs,
            error_jobs=error_jobs,
            total_elapsed_sec=total_elapsed_sec,
            note="existing_summary_detected",
        )

    def _run_one_shard(self, shard_index: int, start: int, end: int) -> CampaignShardRecord:
        outdir = self._shard_outdir(start, end)
        outdir.mkdir(parents=True, exist_ok=True)
        summary_path = outdir / "summary.json"

        if outdir.exists() and summary_path.exists() and not self.force_restart_current:
            record = self._infer_existing_done(outdir)
            if record is not None:
                record.shard_index = shard_index
                record.start = start
                record.end = end
                return record

        cmd = [
            self.python_exe,
            str(self.runner),
            "--manifest",
            str(self.manifest),
            "--outdir",
            str(outdir),
            "--start",
            str(start),
            "--end",
            str(end),
        ]

        started_at = utc_now_iso()
        t0 = time.perf_counter()
        if self.dry_run:
            finished_at = utc_now_iso()
            return CampaignShardRecord(
                shard_index=shard_index,
                start=start,
                end=end,
                outdir=str(outdir),
                status="dry_run",
                started_at_utc=started_at,
                finished_at_utc=finished_at,
                elapsed_sec=0.0,
                exit_code=0,
                summary_path=str(summary_path),
                processed_jobs=None,
                results_count=None,
                no_trade_jobs=None,
                error_jobs=None,
                total_elapsed_sec=None,
                note=" ".join(cmd),
            )

        exit_code = subprocess.run(cmd, check=False).returncode
        elapsed_sec = round(time.perf_counter() - t0, 3)
        finished_at = utc_now_iso()
        summary = self._read_summary(summary_path)
        status = "done" if exit_code == 0 else "error"

        return CampaignShardRecord(
            shard_index=shard_index,
            start=start,
            end=end,
            outdir=str(outdir),
            status=status,
            started_at_utc=started_at,
            finished_at_utc=finished_at,
            elapsed_sec=elapsed_sec,
            exit_code=exit_code,
            summary_path=str(summary_path) if summary_path.exists() else None,
            processed_jobs=int(summary.get("processed_jobs") or 0) if summary else None,
            results_count=int(summary.get("results_count") or 0) if summary else None,
            no_trade_jobs=int(summary.get("no_trade_jobs") or 0) if summary else None,
            error_jobs=int(summary.get("error_jobs") or 0) if summary else None,
            total_elapsed_sec=float(summary.get("total_elapsed_sec") or 0.0) if summary else None,
            note=None if exit_code == 0 else "runner_exit_nonzero",
        )

    def run(self) -> int:
        print("=" * 120)
        print(f"[CAMPAIGN-START] version={VERSION}")
        print(f"[CAMPAIGN-START] manifest={self.manifest}")
        print(f"[CAMPAIGN-START] runner={self.runner}")
        print(f"[CAMPAIGN-START] outroot={self.outroot}")
        print(f"[CAMPAIGN-START] shard_size={self.shard_size}")
        print(f"[CAMPAIGN-START] total_jobs={self.total_jobs}")
        print(f"[CAMPAIGN-START] next_start={self.state.next_start}")
        print("=" * 120)

        self._save_state()

        shard_count = 0
        current_start = self.start_override if self.start_override is not None else self.state.next_start
        global_end = self.end_override if self.end_override is not None else self.total_jobs
        global_end = min(global_end, self.total_jobs)

        while current_start < global_end:
            if self.max_shards is not None and shard_count >= self.max_shards:
                self.state.last_status = "paused_max_shards_reached"
                self.state.next_start = current_start
                self._save_state()
                self._write_campaign_summary()
                print(f"[CAMPAIGN-STOP] reason=max_shards_reached next_start={current_start}")
                return 0

            current_end = min(current_start + self.shard_size, global_end)
            shard_index = shard_count
            print("-" * 120)
            print(f"[SHARD-START] index={shard_index} start={current_start} end={current_end}")
            record = self._run_one_shard(shard_index=shard_index, start=current_start, end=current_end)
            self._append_history(record)

            self.state.last_status = record.status
            self.state.last_outdir = record.outdir
            self.state.last_summary_path = record.summary_path

            if record.status in {"done", "skipped_existing", "dry_run"}:
                self.state.finished_shards += 1
                self.state.next_start = current_end
                self._save_state()
                self._write_campaign_summary()
                print(
                    f"[SHARD-DONE] index={shard_index} status={record.status} "
                    f"processed_jobs={record.processed_jobs} results_count={record.results_count} "
                    f"no_trade_jobs={record.no_trade_jobs} error_jobs={record.error_jobs} outdir={record.outdir}"
                )
                current_start = current_end
                shard_count += 1
                if self.sleep_sec > 0 and current_start < global_end:
                    time.sleep(self.sleep_sec)
                continue

            self.state.next_start = current_start
            self._save_state()
            self._write_campaign_summary()
            print(
                f"[SHARD-ERROR] index={shard_index} exit_code={record.exit_code} "
                f"outdir={record.outdir} summary={record.summary_path}"
            )
            if self.stop_on_error:
                print("[CAMPAIGN-STOP] reason=stop_on_error")
                return int(record.exit_code or 1)

            current_start = current_end
            shard_count += 1

        self.state.last_status = "completed"
        self.state.next_start = global_end
        self._save_state()
        self._write_campaign_summary()
        print("=" * 120)
        print(f"[CAMPAIGN-DONE] version={VERSION}")
        print(f"[CAMPAIGN-DONE] next_start={self.state.next_start}")
        print(f"[CAMPAIGN-DONE] history={self.campaign_history_path}")
        print(f"[CAMPAIGN-DONE] state={self.campaign_state_path}")
        print(f"[CAMPAIGN-DONE] summary={self.campaign_summary_path}")
        print("=" * 120)
        return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run full discovery campaign by orchestrating shard-based research jobs with auto-save and auto-resume."
    )
    parser.add_argument("--manifest", required=True, help="Path to research_job_manifest_full_discovery.jsonl")
    parser.add_argument("--runner", required=True, help="Path to run_v2_fast_research_shard.py")
    parser.add_argument("--outroot", required=True, help="Root output directory for shard folders and campaign state")
    parser.add_argument("--shard-size", type=int, default=1000, help="Number of jobs per shard")
    parser.add_argument("--start", type=int, default=None, help="Optional starting manifest row index override")
    parser.add_argument("--end", type=int, default=None, help="Optional ending manifest row index override")
    parser.add_argument("--max-shards", type=int, default=None, help="Optional max shard count for this invocation")
    parser.add_argument("--sleep-sec", type=float, default=0.0, help="Optional sleep seconds between shards")
    parser.add_argument("--python-exe", default=DEFAULT_PYTHON, help="Python executable used to launch the shard runner")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop the campaign immediately if a shard exits non-zero")
    parser.add_argument(
        "--force-restart-current",
        action="store_true",
        help="Do not skip shard folders that already have summary.json; rerun them instead",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned shards into history without launching the runner")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        orchestrator = CampaignOrchestrator(args)
        return orchestrator.run()
    except CampaignError as exc:
        print(f"[ERROR] version={VERSION} reason={exc}")
        return 2
    except KeyboardInterrupt:
        print(f"[STOPPED] version={VERSION} reason=keyboard_interrupt")
        return 130
    except Exception as exc:
        print(f"[ERROR] version={VERSION} reason=unhandled_exception detail={exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
'''
path = Path('/mnt/data/run_full_discovery_campaign.py')
path.write_text(code, encoding='utf-8')
print(path)
