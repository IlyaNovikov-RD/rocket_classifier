"""Guard against stale hard-coded counts in project documentation.

If you add or remove a test and CI fails here, update the test counts in
CLAUDE.md and README.md to match, then re-commit.
"""

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = ROOT / "tests"


def _count_tests() -> int:
    """Count ``test_*`` functions/methods across all test modules via AST."""
    total = 0
    for path in sorted(TESTS_DIR.glob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                total += 1
    return total


def test_documented_test_count_matches_actual() -> None:
    """CLAUDE.md and README.md test counts must equal the real test count.

    This prevents the '109 → 110' stale-reference problem: whenever a test
    is added or removed the build will fail until the docs are updated.
    """
    actual = _count_tests()

    claude_md = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    m = re.search(r"→\s*(\d+)/\1\)", claude_md)
    assert m, "Could not find test count pattern (e.g. '→ 111/111)') in CLAUDE.md"
    assert int(m.group(1)) == actual, (
        f"CLAUDE.md says {m.group(1)} tests but there are {actual}. Update the count in CLAUDE.md."
    )

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    m_suite = re.search(r"full test suite\s*—\s*(\d+)\s+tests", readme)
    assert m_suite, "Could not find 'full test suite — N tests' in README.md"
    assert int(m_suite.group(1)) == actual, (
        f"README.md says {m_suite.group(1)} tests but there are {actual}. "
        "Update the count in README.md."
    )

    m_table = re.search(r"\*\*(\d+)\s+unit\s*\+\s*contract\s+tests\*\*", readme)
    assert m_table, "Could not find '**N unit + contract tests**' in README.md"
    assert int(m_table.group(1)) == actual, (
        f"README.md table says {m_table.group(1)} tests but there are {actual}. "
        "Update the count in README.md."
    )
