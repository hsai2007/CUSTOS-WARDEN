"""M6: 'all nine output files are written'. Runs only after results/ has
been generated (python -m warden.generate_results) since item 7
(scene_agents) requires M7, built last per the project's build order."""
import pytest

from warden.generate_results import RESULTS_DIR

REQUIRED_OUTPUT_FILES = [
    "metrics_table.png", "metrics_table.csv",         # item 1
    "confusion_matrix.png",                            # item 2
    "per_class_f1.png",                                # item 3
    "cost_breakdown.png",                               # item 4
    "false_facts_surviving.png",                        # item 5
    "scene_demo.txt", "scene_demo.png",                # item 6
    "scene_agents.txt", "scene_agents.png",            # item 7 (M7 only)
    "writes.csv",                                       # item 8
    "module_status.png",                                # item 9
]


@pytest.mark.parametrize("filename", REQUIRED_OUTPUT_FILES)
def test_required_output_file_exists_and_nonempty(filename):
    path = RESULTS_DIR / filename
    assert path.exists(), f"missing required output file: {filename}"
    assert path.stat().st_size > 0, f"required output file is empty: {filename}"


def test_writes_csv_has_frozen_schema():
    import pandas as pd
    from warden.harness import LOG_COLUMNS

    df = pd.read_csv(RESULTS_DIR / "writes.csv")
    assert tuple(df.columns) == LOG_COLUMNS


def test_writes_csv_contains_all_three_configs():
    import pandas as pd
    from warden.harness import ALL_CONFIGS

    df = pd.read_csv(RESULTS_DIR / "writes.csv")
    assert set(df["config"].unique()) == set(ALL_CONFIGS)
