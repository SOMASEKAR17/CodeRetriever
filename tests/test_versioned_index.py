import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from prism.chunker import Chunker
from prism.embedder import HashEmbedder
from prism.engine import Engine
from prism.hashing import git_blob_sha

V1_UTILS = '''def parse_config(path):
    with open(path) as handle:
        return handle.read().split("\\n")


def compute_total(items):
    return sum(item.price for item in items)
'''

V2_UTILS = '''def parse_config(path):
    with open(path) as handle:
        return handle.read().split("\\n")


def compute_total(items, discount):
    subtotal = sum(item.price for item in items)
    return subtotal * (1 - discount)
'''

MAIN = '''from utils import compute_total


def checkout(cart):
    return compute_total(cart.items)
'''


def git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


class VersionedIndexTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.repo = root / "repo"
        self.repo.mkdir()
        git(self.repo, "init", "-q", "-b", "main")
        git(self.repo, "config", "user.email", "test@example.com")
        git(self.repo, "config", "user.name", "Test")
        (self.repo / "utils.py").write_text(V1_UTILS)
        (self.repo / "main.py").write_text(MAIN)
        (self.repo / "README.md").write_text("docs")
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-q", "-m", "v1")
        git(self.repo, "tag", "v1")
        (self.repo / "utils.py").write_text(V2_UTILS)
        git(self.repo, "commit", "-q", "-am", "v2")
        git(self.repo, "tag", "v2")
        git(self.repo, "mv", "main.py", "app.py")
        git(self.repo, "commit", "-q", "-m", "v3")
        git(self.repo, "tag", "v3")
        self.engine = Engine(str(root / "index"), HashEmbedder())
        self.source = self.engine.add_git(str(self.repo), "demo")

    def tearDown(self):
        for source in self.engine._sources.values():
            if hasattr(source, "close"):
                source.close()
        self.engine.store.close()
        self.tmp.cleanup()

    def test_blob_sha_matches_git(self):
        path = self.repo / "utils.py"
        expected = subprocess.run(["git", "hash-object", str(path)], capture_output=True, text=True, check=True).stdout.strip()
        self.assertEqual(git_blob_sha(path.read_bytes()), expected)

    def test_incremental_indexing_reuses_unchanged_code(self):
        stats = {s.label: s for s in self.engine.index(self.source, ["v1", "v2", "v3"], progress=lambda m: None)}
        self.assertEqual(stats["v1"].files, 2)
        self.assertEqual(stats["v1"].chunks_embedded, 4)
        self.assertEqual(stats["v2"].blobs_reused, 1)
        self.assertEqual(stats["v2"].chunks_embedded, 1)
        self.assertEqual(stats["v3"].blobs_new, 0)
        self.assertEqual(stats["v3"].chunks_embedded, 0)

    def test_search_respects_versions(self):
        self.engine.index(self.source, ["v1", "v2", "v3"], progress=lambda m: None)
        query = "compute total with discount subtotal"
        v1 = self.engine.search(query, self.source, "v1", k=1)[0]
        v2 = self.engine.search(query, self.source, "v2", k=1)[0]
        self.assertNotIn("discount", v1.code)
        self.assertIn("discount", v2.code)
        self.assertEqual({o["version_id"] for o in v1.occurrences}, {self.engine.resolve_version(self.source, "v1")})
        v3_paths = {o["path"] for o in self.engine.search("checkout cart", self.source, "v3", k=1)[0].occurrences}
        self.assertEqual(v3_paths, {"app.py"})
        everywhere = self.engine.search(query, self.source, "all", k=1)[0]
        self.assertEqual(len({o["version_id"] for o in everywhere.occurrences}), 2)

    def test_folder_and_corpus_sources(self):
        folder = self.engine.add_folder(str(self.repo), "folder")
        stats = self.engine.index(folder, progress=lambda m: None)[0]
        self.assertEqual(stats.files, 2)
        corpus_file = Path(self.tmp.name) / "corpus_v1.jsonl"
        corpus_file.write_text("\n".join(json.dumps({"_id": f"d{i}", "text": f"print({i})"}) for i in range(3)))
        corpus = self.engine.add_corpus([str(corpus_file)], "apps")
        stats = self.engine.index(corpus, progress=lambda m: None)[0]
        self.assertEqual(stats.files, 3)


if __name__ == "__main__":
    unittest.main()
