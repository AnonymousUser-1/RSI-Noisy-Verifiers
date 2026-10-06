"""Which code a run executed: git_commit, git_dirty and a source snapshot.

RUN_COST_SPEC 1 binds run.json to `git_commit` (the full SHA, or null with a
reason, never a branch name), `source_hash` (rsi.common.source_hash, kept as it
is) and `git_dirty` (with the path and hash of a source snapshot when true).
`source_hash` fingerprints the files that ran, but it cannot name their commit
or say whether they differ from it; the helpers here answer both, for the same
set of files.

rsi/common.py is not changed for this.  `source_files` and `tree_hash` repeat
rsi.common.source_hash's file list and hash with the tree as an argument, so a
test can point them at a scratch repository; tests/test_provenance.py checks
that at this checkout `tree_hash()` equals `rsi.common.source_hash()`.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .common import digest, file_hash

SOURCE_ROOT = Path(__file__).resolve().parents[1]


def source_files(root=SOURCE_ROOT):
    """The files rsi.common.source_hash() reads, for the tree at `root`."""
    root = Path(root)
    # rsi.common.source_hash's globs: no .venv, model cache or experiment output.
    return sorted(list(root.glob("*.py")) + list((root / "rsi").rglob("*.py")) + list((root / "tests").rglob("*.py")))


def tree_hash(root=SOURCE_ROOT):
    """rsi.common.source_hash() of the tree at `root`."""
    root = Path(root)
    return digest({str(p.relative_to(root)): file_hash(p) for p in source_files(root)})


def is_source_path(path):
    """Whether `path`, POSIX and relative to the root, is a file source_files() lists."""
    return path.endswith(".py") and ("/" not in path or path.split("/", 1)[0] in ("rsi", "tests"))


def git_state(root=SOURCE_ROOT):
    """`git_commit` and `git_dirty` for the tree at `root`.

    The commit is `git rev-parse HEAD`.  Dirty means the files source_files()
    lists are not exactly HEAD's files: one is modified, staged, new (untracked
    or ignored) or deleted.  So False is a statement about the code that ran,
    not about the index.  When git cannot answer, both are None and `reason`
    says why.
    """
    root = Path(root)

    def git(*args, stdin=None):
        return subprocess.run(["git", "--no-optional-locks", "-C", str(root), *args], input=stdin,
                              capture_output=True, text=True, encoding="utf-8", errors="replace",
                              check=True).stdout

    files = [p.relative_to(root).as_posix() for p in source_files(root)]
    try:
        commit = git("rev-parse", "--verify", "HEAD").strip()
        committed = {}
        for entry in git("ls-tree", "-r", "-z", "HEAD").split("\0"):
            if entry:
                info, path = entry.split("\t", 1)
                if is_source_path(path):
                    committed[path] = info.split()[2]
        # hash-object applies the same clean filters as `git add`, so an
        # unchanged file hashes to its committed blob whatever its line endings.
        blobs = git("hash-object", "--stdin-paths", stdin="".join(f + "\n" for f in files)).split() if files else []
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = (getattr(exc, "stderr", None) or str(exc)).strip()
        return {"git_commit": None, "git_dirty": None, "changes": None,
                "reason": "git could not describe %s: %s" % (root, detail)}
    running = dict(zip(files, blobs))
    changes = sorted(p for p in set(committed) | set(running) if committed.get(p) != running.get(p))
    return {"git_commit": commit, "git_dirty": bool(changes), "changes": changes, "reason": None}


def snapshot_source(dest, root=SOURCE_ROOT):
    """Copy the files source_files() lists into `dest` and return the copy's tree_hash."""
    root, dest = Path(root), Path(dest)
    for path in source_files(root):
        target = dest / path.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    return tree_hash(dest)
