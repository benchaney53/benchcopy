import json
from pathlib import Path
import tempfile
import shutil
import unittest

import job_book_tool as jbt


class JobBookToolTestCase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "config.json"
        self.sample_root = Path(self.temp_dir) / "root"
        self.sample_root.mkdir()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_save_and_load_configuration(self):
        config = jbt.JobBookConfig(search_root=str(self.sample_root))
        config.add_rule("*.md")
        config.add_rule("*.txt", enabled=False)
        jbt.save_config(config, self.config_path)

        loaded = jbt.load_config(self.config_path)
        self.assertEqual(loaded.search_root, str(self.sample_root))
        self.assertEqual(len(loaded.file_rules), 2)
        self.assertTrue(any(rule.pattern == "*.md" and rule.enabled for rule in loaded.file_rules))
        self.assertTrue(any(rule.pattern == "*.txt" and not rule.enabled for rule in loaded.file_rules))

    def test_scan_uses_enabled_patterns(self):
        config = jbt.JobBookConfig(search_root=str(self.sample_root))
        config.set_rule_state("*.md", True)
        config.set_rule_state("*.log", False)

        markdown_file = self.sample_root / "notes.md"
        markdown_file.write_text("hello")
        (self.sample_root / "app.log").write_text("log")

        matches = jbt.scan_files(config)
        self.assertIn(markdown_file.resolve(), matches)
        self.assertNotIn((self.sample_root / "app.log").resolve(), matches)

    def test_format_config_prints_states(self):
        config = jbt.JobBookConfig(search_root=str(self.sample_root))
        config.set_rule_state("*.md", True)
        config.set_rule_state("*.log", False)
        output = jbt.format_config(config)
        self.assertIn("Search root", output)
        self.assertIn("*.md (enabled)", output)
        self.assertIn("*.log (disabled)", output)


if __name__ == "__main__":
    unittest.main()
