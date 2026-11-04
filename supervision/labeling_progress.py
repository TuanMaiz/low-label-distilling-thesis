"""Progress bar and log file for teacher-labelling passes.

``LabelingProgress`` is passed as ``progress=`` to ``screening_lib.run_setting``.
It only observes: the bar (tqdm, on stderr) shows pairs done, spend so far and
the share of matches; the log file (``progress.log`` in the pass folder) gets one
line at start and finish, every error, and a summary every ``log_every`` pairs.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except ImportError:   # optional: without tqdm a plain line is printed every log_every pairs
    tqdm = None


class LabelingProgress:
    def __init__(self, log_path: Path, label: str, log_every: int = 100, show_bar: bool = True) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(f"labeling.{label}.{id(self)}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        self._handler = logging.FileHandler(log_path, encoding="utf-8")
        self._handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        self.logger.addHandler(self._handler)
        self.label, self.log_every, self.show_bar = label, log_every, show_bar
        self.bar: Any = None
        self.matches = self.new = self.errors = 0

    def __call__(self, event: dict[str, Any]) -> None:
        kind, done, total, cost = event["event"], event["done"], event["total"], event["accounted_cost_usd"]
        if kind == "start":
            self.logger.info("%s start: %d/%d already done, spent USD %.4f, ceiling USD %.2f",
                             self.label, done, total, cost, event["spend_ceiling_usd"])
            if tqdm is not None:
                self.bar = tqdm(total=total, initial=done, desc=self.label, unit="pair", disable=not self.show_bar)
            elif self.show_bar:
                print(f"{self.label}: {done}/{total} done (install tqdm for a progress bar)", file=sys.stderr)
        elif kind == "valid":
            self.new += 1
            self.matches += event["result"] == "match"
            if self.bar is not None:
                self.bar.update(1)
                self.bar.set_postfix(usd=f"{cost:.3f}", match=f"{self.matches / self.new:.0%}", errors=self.errors)
            if done % self.log_every == 0 or done == total:
                if self.bar is None and self.show_bar:
                    print(f"{self.label}: {done}/{total} done, USD {cost:.3f}, errors {self.errors}", file=sys.stderr)
                self.logger.info("%s %d/%d done, spent USD %.4f, match share this run %.1f%%, errors %d",
                                 self.label, done, total, cost, 100 * self.matches / self.new, self.errors)
        elif kind == "error":
            self.errors += 1
            message = (f"{self.label} error on {event['pair_id']} attempt {event['attempt']} "
                       f"({event['error_type']}, retryable={event['retryable']}): {event['error']}")
            self.logger.warning(message)
            if self.bar is not None:
                self.bar.write(message)
            elif self.show_bar:
                print(message, file=sys.stderr)
        elif kind == "finish":
            self.logger.info("%s finished: %d/%d, spent USD %.4f, errors %d, result %s",
                             self.label, done, total, cost, self.errors, event["final_path"])
            self.close()

    def close(self) -> None:
        if self.bar is not None:
            self.bar.close()
            self.bar = None
        self.logger.removeHandler(self._handler)
        self._handler.close()
