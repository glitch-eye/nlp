import tempfile
import unittest
from pathlib import Path

from nlp.app.knowledge_base import DATA_FILES, KnowledgeBase, default_data_dir


class RepositoryDatasetTests(unittest.TestCase):
    def test_default_data_dir_points_to_bundled_python_data(self):
        data_dir = default_data_dir()
        self.assertEqual(data_dir.name, "data")
        self.assertEqual(data_dir.parent.name, "python")
        self.assertTrue((data_dir / "entities.txt").is_file())
        for filename in DATA_FILES:
            self.assertTrue((data_dir / filename).is_file(), filename)

    def test_default_knowledge_base_loads_all_source_types(self):
        kb = KnowledgeBase(default_data_dir())
        loaded_kinds = {document["kind"] for document in kb.documents}
        self.assertEqual(loaded_kinds, set(DATA_FILES.values()))
        self.assertTrue(kb.aliases)
        self.assertTrue(kb.documents)
        self.assertEqual(kb.data_dir, default_data_dir().resolve())

    def test_incomplete_dataset_fails_instead_of_silently_loading_partial_data(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "Missing"):
                KnowledgeBase(Path(directory))


if __name__ == "__main__":
    unittest.main()
