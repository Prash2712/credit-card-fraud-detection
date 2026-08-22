import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_loader import load_data  # noqa: E402


def test_load_data_reads_existing_csv(tmp_path):
    source = tmp_path / "creditcard.csv"
    expected = pd.DataFrame({"V1": [0.1, 0.2], "Amount": [10.0, 20.0], "Class": [0, 1]})
    expected.to_csv(source, index=False)

    actual = load_data(str(source))

    pd.testing.assert_frame_equal(actual, expected)


def test_load_data_fails_clearly_for_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Dataset not found"):
        load_data(str(tmp_path / "missing.csv"))
