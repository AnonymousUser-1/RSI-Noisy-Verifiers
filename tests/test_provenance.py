"""rsi.provenance: the git_commit, git_dirty and source snapshot that run.json binds.

RUN_COST_SPEC 1: git_commit is the full SHA or null with a reason; git_dirty is
true, false or null; when it is not false the run keeps a snapshot of the
source it ran.  The scratch repositories below are built with git itself, and
every expected answer follows from what the test did to the tree, not from
rsi.provenance.
"""
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rsi.common import file_hash, source_hash
from rsi.provenance import SOURCE_ROOT, git_state, snapshot_source, source_files, tree_hash


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True).stdout


class SourceListTests(unittest.TestCase):
    def test_tree_hash_is_source_hash_at_this_checkout(self):
        """The drift guard: provenance repeats rsi.common.source_hash rather than changing it."""
        self.assertEqual(tree_hash(), source_hash())

    def test_source_files_are_the_three_globs(self):
        expected = sorted(list(SOURCE_ROOT.glob("*.py")) + list((SOURCE_ROOT / "rsi").rglob("*.py"))
                          + list((SOURCE_ROOT / "tests").rglob("*.py")))
        self.assertEqual(source_files(), expected)
        self.assertIn(SOURCE_ROOT / "tests" / "test_provenance.py", source_files())


@unittest.skipUnless(shutil.which("git"), "git is not installed")
class GitStateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "repo"
        (self.root / "rsi").mkdir(parents=True)
        (self.root / "tests").mkdir()
        (self.root / "scripts").mkdir()
        (self.root / "entry.py").write_bytes(b"print('entry')\n")
        (self.root / "rsi" / "core.py").write_bytes(b"VALUE = 1\n")
        (self.root / "tests" / "test_x.py").write_bytes(b"pass\n")
        (self.root / "scripts" / "tool.py").write_bytes(b"pass\n")
        (self.root / "README.md").write_bytes(b"readme\n")
        git(self.root, "init", "-q")
        git(self.root, "config", "user.name", "provenance test")
        git(self.root, "config", "user.email", "provenance@example.invalid")
        git(self.root, "config", "core.autocrlf", "true")
        git(self.root, "add", "-A")
        git(self.root, "commit", "-q", "-m", "base")
        self.head = git(self.root, "rev-parse", "HEAD").strip()

    def tearDown(self):
        self.tmp.cleanup()

    def assertState(self, dirty, changes):
        state = git_state(self.root)
        self.assertEqual(state["git_commit"], self.head)
        self.assertEqual(len(state["git_commit"]), 40)
        self.assertIs(state["git_dirty"], dirty)
        self.assertEqual(state["changes"], changes)
        self.assertIsNone(state["reason"])

    def test_clean_tree(self):
        self.assertState(False, [])

    def test_modified_source_is_dirty(self):
        (self.root / "rsi" / "core.py").write_bytes(b"VALUE = 2\n")
        self.assertState(True, ["rsi/core.py"])

    def test_untracked_source_is_dirty(self):
        (self.root / "rsi" / "extra.py").write_bytes(b"EXTRA = 1\n")
        self.assertState(True, ["rsi/extra.py"])

    def test_deleted_source_is_dirty(self):
        (self.root / "tests" / "test_x.py").unlink()
        self.assertState(True, ["tests/test_x.py"])

    def test_staged_change_is_dirty(self):
        (self.root / "entry.py").write_bytes(b"print('changed')\n")
        git(self.root, "add", "entry.py")
        self.assertState(True, ["entry.py"])

    def test_staged_then_restored_file_is_clean(self):
        """Dirty is about the code that ran: the index alone does not make it dirty."""
        (self.root / "entry.py").write_bytes(b"print('changed')\n")
        git(self.root, "add", "entry.py")
        (self.root / "entry.py").write_bytes(b"print('entry')\n")
        self.assertState(False, [])

    def test_files_outside_the_source_list_are_not_dirty(self):
        (self.root / "README.md").write_bytes(b"changed\n")
        (self.root / "notes.txt").write_bytes(b"new\n")
        (self.root / "scripts" / "tool.py").write_bytes(b"changed = True\n")
        self.assertState(False, [])

    def test_line_endings_alone_are_not_dirty(self):
        (self.root / "entry.py").write_bytes(b"print('entry')\r\n")
        self.assertState(False, [])

    def test_outside_a_repository(self):
        plain = Path(self.tmp.name) / "plain"
        (plain / "rsi").mkdir(parents=True)
        (plain / "entry.py").write_bytes(b"pass\n")
        with mock.patch.dict(os.environ, {"GIT_CEILING_DIRECTORIES": self.tmp.name}):
            state = git_state(plain)
        self.assertIsNone(state["git_commit"])
        self.assertIsNone(state["git_dirty"])
        self.assertIsNone(state["changes"])
        self.assertIn(str(plain), state["reason"])

    def test_snapshot_holds_exactly_the_source_files(self):
        (self.root / "rsi" / "core.py").write_bytes(b"VALUE = 3\n")
        dest = Path(self.tmp.name) / "snapshot"
        copied = snapshot_source(dest, self.root)
        listed = sorted(p.relative_to(self.root).as_posix() for p in source_files(self.root))
        self.assertEqual(listed, ["entry.py", "rsi/core.py", "tests/test_x.py"])
        self.assertEqual(sorted(p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file()), listed)
        for name in listed:
            self.assertEqual(file_hash(dest / name), file_hash(self.root / name))
        self.assertEqual(copied, tree_hash(self.root))


if __name__ == "__main__":
    unittest.main()
