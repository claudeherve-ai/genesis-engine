"""Tests for TestResults and TestFailure models."""

import pytest
from pydantic import ValidationError

from genesis.models.test_results import TestResults, TestFailure


# ── TestFailure ─────────────────────────────────────────────────────────────

class TestTestFailure:
    def test_creation(self):
        tf = TestFailure(
            scenario="refund_request",
            agent="billing_agent",
            expected="Refund processed",
            actual="Refund failed: insufficient funds",
            metric="accuracy",
        )
        assert tf.scenario == "refund_request"
        assert tf.agent == "billing_agent"
        assert tf.expected == "Refund processed"
        assert tf.actual == "Refund failed: insufficient funds"
        assert tf.metric == "accuracy"

    def test_empty_strings_ok(self):
        tf = TestFailure(
            scenario="", agent="", expected="", actual="", metric="",
        )
        assert tf.scenario == ""
        assert tf.agent == ""
        assert tf.expected == ""
        assert tf.actual == ""
        assert tf.metric == ""

    def test_fields_required(self):
        with pytest.raises(ValidationError):
            TestFailure(scenario="s", agent="a", expected="e")
        with pytest.raises(ValidationError):
            TestFailure(scenario="s", agent="a", expected="e", actual="")

    def test_model_dump(self):
        tf = TestFailure(
            scenario="s", agent="a", expected="e", actual="a2", metric="m",
        )
        d = tf.model_dump()
        assert d == {
            "scenario": "s", "agent": "a", "expected": "e",
            "actual": "a2", "metric": "m",
        }


# ── TestResults ─────────────────────────────────────────────────────────────

class TestTestResults:
    def test_defaults(self):
        tr = TestResults()
        assert tr.scenarios_run == 0
        assert tr.scenarios_passed == 0
        assert tr.overall_score == 0.0
        assert tr.metrics == {}
        assert tr.failures == []
        assert tr.passed is False

    def test_full_creation(self):
        failures = [
            TestFailure(scenario="s1", agent="a1", expected="e1", actual="a1", metric="m"),
            TestFailure(scenario="s2", agent="a2", expected="e2", actual="a2", metric="m"),
        ]
        tr = TestResults(
            scenarios_run=10,
            scenarios_passed=8,
            overall_score=0.85,
            metrics={"accuracy": 0.9, "latency": 0.8},
            failures=failures,
            passed=False,
        )
        assert tr.scenarios_run == 10
        assert tr.scenarios_passed == 8
        assert tr.overall_score == 0.85
        assert tr.metrics == {"accuracy": 0.9, "latency": 0.8}
        assert len(tr.failures) == 2
        assert tr.failure_count == 2

    def test_pass_rate_zero_runs(self):
        tr = TestResults(scenarios_run=0, scenarios_passed=0)
        assert tr.pass_rate == 0.0

    def test_pass_rate_calculation(self):
        tr = TestResults(scenarios_run=10, scenarios_passed=7)
        assert tr.pass_rate == 0.7

    def test_pass_rate_all_passed(self):
        tr = TestResults(scenarios_run=5, scenarios_passed=5)
        assert tr.pass_rate == 1.0

    def test_pass_rate_none_passed(self):
        tr = TestResults(scenarios_run=5, scenarios_passed=0)
        assert tr.pass_rate == 0.0

    def test_failure_count_empty(self):
        tr = TestResults(failures=[])
        assert tr.failure_count == 0

    def test_failure_count(self):
        fs = [TestFailure(scenario=f"s{i}", agent=f"a{i}",
                          expected="e", actual="a", metric="m")
              for i in range(3)]
        tr = TestResults(failures=fs)
        assert tr.failure_count == 3

    def test_check_threshold_meets(self):
        tr = TestResults(overall_score=0.85)
        result = tr.check_threshold(threshold=0.80)
        assert result is True
        assert tr.passed is True

    def test_check_threshold_meets_exactly(self):
        tr = TestResults(overall_score=0.80)
        result = tr.check_threshold(threshold=0.80)
        assert result is True
        assert tr.passed is True

    def test_check_threshold_fails(self):
        tr = TestResults(overall_score=0.75)
        result = tr.check_threshold(threshold=0.80)
        assert result is False
        assert tr.passed is False

    def test_check_threshold_default_is_0_80(self):
        tr = TestResults(overall_score=0.90)
        tr.check_threshold()
        assert tr.passed is True

    def test_overall_score_boundaries(self):
        TestResults(overall_score=0.0)
        TestResults(overall_score=1.0)
        with pytest.raises(ValidationError):
            TestResults(overall_score=-0.01)
        with pytest.raises(ValidationError):
            TestResults(overall_score=1.01)

    def test_scenarios_non_negative(self):
        with pytest.raises(ValidationError):
            TestResults(scenarios_run=-1)
        with pytest.raises(ValidationError):
            TestResults(scenarios_passed=-1)

    def test_metrics_default_is_dict(self):
        tr = TestResults()
        assert isinstance(tr.metrics, dict)
        assert tr.metrics == {}

    def test_model_dump(self):
        tr = TestResults(scenarios_run=1, scenarios_passed=1, overall_score=1.0)
        d = tr.model_dump()
        assert d["scenarios_run"] == 1
        assert d["scenarios_passed"] == 1
        assert d["overall_score"] == 1.0
        assert d["passed"] is False
        assert d["failures"] == []

    def test_failures_default_is_empty_list(self):
        tr = TestResults()
        assert tr.failures == []

    def test_pass_rate_integer_division_avoided(self):
        tr = TestResults(scenarios_run=3, scenarios_passed=2)
        assert tr.pass_rate == pytest.approx(2 / 3)
