"""Smart Tagging loader on a small synthetic SportsCode-style XML (no competition data), and its guard.

    idsse-tests\\.venv\\Scripts\\python -m tests.test_smart_tagging
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from possession import smart_tagging as st  # noqa: E402

XML = """<?xml version="1.0"?>
<file><ALL_INSTANCES>
<instance><ID>1</ID><start>447.4</start><end>452.4</end><code>SET PIECES</code>
 <label><group>Team</group><text>Barcelona</text></label><label><group>Half</group><text>1st Half</text></label>
 <label><group>Type</group><text>Kick Off (Start)</text></label></instance>
<instance><ID>2</ID><start>100.04</start><end>104.0</end><code>SET PIECES</code>
 <label><group>Team</group><text>N/A</text></label><label><group>Type</group><text>Throw-in</text></label></instance>
<instance><ID>3</ID><start>200.0</start><end>203.0</end><code>Set Pieces</code>
 <label><group>Team</group><text>ATLETIC CLUB</text></label><label><group>Type</group><text>corner kick</text></label></instance>
<instance><ID>4</ID><start>300.0</start><end>310.0</end><code>BUILD UP</code>
 <label><group>Team</group><text>Barcelona</text></label></instance>
<instance><ID>5</ID><start>0.018</start><end>3.0</end><code>SET PIECES</code>
 <label><group>Team</group><text>Atletic Club</text></label><label><group>Type</group><text>Free Kick</text></label></instance>
</ALL_INSTANCES></file>
"""


class T(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.p = Path(self.d.name) / "synthetic_tagging.xml"
        self.p.write_text(XML)

    def tearDown(self):
        self.d.cleanup()

    def test_parse(self):
        rs = st.parse(self.p)
        self.assertEqual([(r["frame"], r["type"], r["team"]) for r in rs],
                         [(0, "FreeKick", "A"), (2501, "ThrowIn", None), (5000, "CornerKick", "A"),
                          (11185, "KickOff", "H")])
        self.assertEqual(rs[-1]["end_frame"], 11310)
        self.assertEqual(rs[-1]["half"], "1st Half")

    def test_to_tags(self):
        tags = st.to_tags(st.parse(self.p))
        self.assertEqual([t["team"] for t in tags], [1, None, 1, 0])

    def test_load_with_path_outside_claude(self):
        old = os.environ.pop("CLAUDECODE", None)
        try:
            self.assertEqual(len(st.load_restarts(self.p)), 4)
        finally:
            if old is not None:
                os.environ["CLAUDECODE"] = old

    def test_refuses_under_claude(self):
        old = os.environ.get("CLAUDECODE")
        os.environ["CLAUDECODE"] = "1"
        try:
            with self.assertRaises(SystemExit):
                st.load_restarts(self.p)
            with self.assertRaises(SystemExit):
                st.load_restarts()
        finally:
            if old is None:
                os.environ.pop("CLAUDECODE")
            else:
                os.environ["CLAUDECODE"] = old


if __name__ == "__main__":
    unittest.main(verbosity=2)
