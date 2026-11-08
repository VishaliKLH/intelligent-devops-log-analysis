"""Tests for the Live Log Check (src/predict.py)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import preprocess  # noqa: E402
import train_model  # noqa: E402
import predict  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def trained():
    if not predict.artifacts_ready():
        preprocess.preprocess()
        train_model.train_and_detect()


NORMAL = "Jun 20 10:15:01 combo su(pam_unix)[21416]: session opened for user cyrus by (uid=0)"


def test_normal_line_is_normal():
    res = predict.analyze_lines(NORMAL)
    assert res.loc[0, "status"] == "Normal"


def test_invalid_format_is_reported():
    res = predict.analyze_lines("hello world")
    assert res.loc[0, "status"] == "Invalid format"


def test_failed_login_burst_is_flagged():
    burst = "\n".join(
        f"Jun 20 03:10:{i:02d} combo sshd(pam_unix)[{500 + i}]: authentication failure; "
        "logname= uid=0 euid=0 tty=NODEVssh ruser= rhost=203.0.113.5  user=root"
        for i in range(20)
    )
    res = predict.analyze_lines(burst)
    assert (res["status"] == "Anomaly").sum() >= 5


def test_blank_input_gives_empty_result():
    assert predict.analyze_lines("  \n ").empty
