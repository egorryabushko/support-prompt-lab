import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import prompt_lab as lab


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.cases = lab.read_json(lab.ROOT / "data" / "cases.json")
        self.answers = {
            case["id"]: {**case["expected"], "reason": "Тестовый ответ."}
            for case in self.cases
        }

    def test_all_correct(self):
        result = lab.evaluate(self.cases, self.answers)
        self.assertEqual(result["passed"], len(self.cases))
        self.assertEqual(result["accuracy"]["category"], 1)

    def test_missing_answer_stays_in_denominator(self):
        del self.answers["01"]
        result = lab.evaluate(self.cases, self.answers)
        self.assertEqual(result["total"], 12)
        self.assertEqual(result["passed"], 11)
        self.assertAlmostEqual(result["accuracy"]["category"], 11 / 12)

    def test_invalid_boolean_is_not_accepted(self):
        for value in ("false", 0, 1, None):
            with self.subTest(value=value):
                answer = {**self.answers["01"], "needs_clarification": value}
                self.assertFalse(lab.valid_prediction(answer))

    def test_wrong_priority_only_reduces_priority_metric(self):
        self.answers["01"]["priority"] = "high"
        result = lab.evaluate(self.cases, self.answers)
        self.assertEqual(result["passed"], 11)
        self.assertEqual(result["accuracy"]["category"], 1)
        self.assertAlmostEqual(result["accuracy"]["priority"], 11 / 12)

    def test_schema_rejects_extra_keys_empty_reason_and_invalid_enum(self):
        for answer in ({**self.answers["01"], "extra": 1},
                       {**self.answers["01"], "reason": " "},
                       {**self.answers["01"], "category": "refund"}, None, []):
            with self.subTest(answer=answer):
                self.assertFalse(lab.valid_prediction(answer))

    def test_unknown_ids_and_duplicate_cases_rejected(self):
        with self.assertRaises(ValueError):
            lab.evaluate(self.cases, {"unknown": self.answers["01"]})
        with self.assertRaises(ValueError):
            lab.evaluate([self.cases[0], self.cases[0]], {})
        with self.assertRaises(ValueError):
            lab.evaluate([], {})

    def test_demo_fixtures_are_complete(self):
        for version, passed in (("v1", 5), ("v2", 11)):
            answers = lab.read_json(lab.ROOT / "examples" / f"{version}.json")
            self.assertEqual(set(answers), set(self.answers))
            result = lab.evaluate(self.cases, answers)
            self.assertEqual(result["valid"], 12)
            self.assertEqual(result["passed"], passed)

    def test_report_identifies_demo_and_records_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            lab.save_report(directory, "demo", lab.ROOT / "data" / "cases.json",
                            {"v1": lab.evaluate(self.cases, self.answers)})
            report = lab.read_json(Path(directory) / "report.json")
            self.assertEqual(report["mode"], "demo")
            self.assertIsNone(report["model"])
            self.assertEqual(len(report["prompt_sha256"]["v1"]), 64)


class ApiTests(unittest.TestCase):
    def test_reads_completed_response(self):
        answer = {"category": "account", "priority": "normal",
                  "needs_clarification": False, "reason": "Не получается войти."}
        response = {"status": "completed", "output": [
            {"type": "reasoning"},
            {"type": "message", "content": [{"type": "output_text", "text": json.dumps(answer)}]},
        ]}
        self.assertEqual(lab.parse_response(response), answer)

    def test_refusal_incomplete_and_empty_response_are_errors(self):
        for response in ({"status": "incomplete"}, {"status": "completed", "output": []},
                         {"status": "completed", "output": [{"type": "message", "content": [
                             {"type": "refusal", "refusal": "refused"}]}]}):
            with self.subTest(response=response), self.assertRaises(ValueError):
                lab.parse_response(response)

    def test_live_payload_keeps_ticket_separate_from_prompt(self):
        answer = {"category": "other", "priority": "normal", "needs_clarification": True, "reason": "Нет текста."}
        response = {"status": "completed", "output": [{"type": "message", "content": [
            {"type": "output_text", "text": json.dumps(answer)}]}]}
        import io
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-placeholder"}), \
                patch("prompt_lab.urlopen", return_value=io.BytesIO(json.dumps(response).encode())) as mock:
            self.assertEqual(lab.classify("", "v2", "test-model"), answer)
            request = mock.call_args.args[0]
            payload = json.loads(request.data)
            self.assertFalse(payload["store"])
            self.assertEqual(payload["input"][1], {"role": "user", "content": ""})
            self.assertTrue(payload["text"]["format"]["strict"])


if __name__ == "__main__":
    unittest.main()
