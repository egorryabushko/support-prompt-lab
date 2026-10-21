"""Небольшая лаборатория промптов. Python 3.10+, без сторонних зависимостей."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
FIELDS = ("category", "priority", "needs_clarification")
SCHEMA = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": ["billing", "technical", "account", "other"]},
        "priority": {"type": "string", "enum": ["low", "normal", "high"]},
        "needs_clarification": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": [*FIELDS, "reason"],
    "additionalProperties": False,
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def valid_prediction(value):
    if not isinstance(value, dict) or set(value) != set(SCHEMA["required"]):
        return False
    return (
        value["category"] in SCHEMA["properties"]["category"]["enum"]
        and value["priority"] in SCHEMA["properties"]["priority"]["enum"]
        and type(value["needs_clarification"]) is bool
        and isinstance(value["reason"], str)
        and bool(value["reason"].strip())
    )


def evaluate(cases, predictions):
    """Пропуски и невалидные ответы считаются ошибками; неизвестные ID отклоняются."""
    ids = [case["id"] for case in cases]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Набор кейсов пуст или содержит повторяющиеся ID.")
    if not isinstance(predictions, dict) or set(predictions) - set(ids):
        raise ValueError("Ответы должны быть объектом с известными ID кейсов.")
    rows = []
    for case in cases:
        actual = predictions.get(case["id"])
        valid = valid_prediction(actual)
        matches = {field: valid and actual[field] == case["expected"][field] for field in FIELDS}
        rows.append({
            "id": case["id"], "text": case["text"], "kind": case["kind"],
            "expected": case["expected"], "actual": actual,
            "valid": valid, "matches": matches, "passed": all(matches.values()),
        })
    total = len(rows)
    return {
        "total": total,
        "valid": sum(row["valid"] for row in rows),
        "passed": sum(row["passed"] for row in rows),
        "accuracy": {field: sum(row["matches"][field] for row in rows) / total for field in FIELDS},
        "rows": rows,
    }


def parse_response(response):
    if response.get("status") != "completed":
        raise ValueError("Ответ модели не завершён.")
    texts = []
    for item in response.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "refusal":
                raise ValueError("Модель отказалась обрабатывать обращение.")
            if content.get("type") == "output_text":
                texts.append(content["text"])
    if not texts:
        raise ValueError("Модель не вернула текстовый ответ.")
    prediction = json.loads("".join(texts))
    if not valid_prediction(prediction):
        raise ValueError("Ответ модели не соответствует контракту.")
    return prediction


def classify(text, version, model):
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("Задайте OPENAI_API_KEY или используйте команду demo.")
    prompt = (ROOT / "prompts" / f"{version}.txt").read_text(encoding="utf-8")
    payload = {
        "model": model, "store": False,
        "input": [{"role": "system", "content": prompt}, {"role": "user", "content": text}],
        "text": {"format": {"type": "json_schema", "name": "ticket", "strict": True, "schema": SCHEMA}},
        "max_output_tokens": 2048,
    }
    request = Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=60) as response:
            return parse_response(json.load(response))
    except HTTPError as error:
        raise ValueError(f"API: HTTP {error.code}. Проверьте ключ, модель, доступ и лимиты.") from None
    except (URLError, TimeoutError):
        raise ValueError("Не удалось получить ответ API: сеть или таймаут.") from None


def show_summary(version, result):
    print(f"{version}: полностью верно {result['passed']}/{result['total']}; "
          f"валидный формат {result['valid']}/{result['total']}")
    for field, accuracy in result["accuracy"].items():
        print(f"  {field}: {accuracy:.1%}")
    for row in result["rows"]:
        if not row["passed"]:
            fields = ", ".join(key for key, matched in row["matches"].items() if not matched)
            print(f"  Ошибка {row['id']}: {fields}")


def save_report(directory, mode, cases_path, results, model=None, errors=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    report = {
        "mode": mode, "model": model,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": hashlib.sha256(cases_path.read_bytes()).hexdigest(),
        "prompt_sha256": {
            version: hashlib.sha256((ROOT / "prompts" / f"{version}.txt").read_bytes()).hexdigest()
            for version in results
        },
        "errors": errors or {}, "results": results,
    }
    write_json(directory / "report.json", report)
    title = "Учебная демонстрация — вручную составленные ответы" if mode == "demo" else "Результаты API"
    lines = [f"# {title}", "", f"Модель: {model or 'не вызывалась'}", "",
             "| Промпт | Полностью верно | Категория | Приоритет | Уточнение | Формат |",
             "|---|---:|---:|---:|---:|---:|"]
    for version, result in results.items():
        scores = " | ".join(f"{result['accuracy'][field]:.1%}" for field in FIELDS)
        lines.append(f"| {version} | {result['passed']}/{result['total']} | {scores} | {result['valid']}/{result['total']} |")
    lines += ["", "Оценка reason ограничена наличием непустой строки; смысл объяснения проверяется вручную.", ""]
    if mode == "demo":
        lines += ["Ответы составлены вручную для демонстрации проверок. Эти числа не измеряют качество LLM и не доказывают преимущество v2.", ""]
    if errors:
        lines += ["Ошибки API сохранены в report.json; такие ответы включены в знаменатель как ошибки.", ""]
    (directory / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Отчёт: {directory / 'report.md'}")


def main():
    parser = argparse.ArgumentParser(description="Сравнение промптов для классификации обращений")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Учебный пример без сети и API-ключа")
    demo.add_argument("--out", type=Path, default=ROOT / "reports" / "demo")
    single = commands.add_parser("classify", help="Классифицировать обращение через API")
    single.add_argument("--text", required=True)
    single.add_argument("--version", choices=["v1", "v2"], default="v2")
    single.add_argument("--model", required=True)
    run = commands.add_parser("evaluate", help="Сравнить обе версии через API: 24 запроса")
    run.add_argument("--model", required=True)
    run.add_argument("--out", type=Path, default=ROOT / "reports" / "live")
    args = parser.parse_args()
    try:
        if args.command == "classify":
            print(json.dumps(classify(args.text, args.version, args.model), ensure_ascii=False, indent=2))
            return 0
        cases_path = ROOT / "data" / "cases.json"
        cases = read_json(cases_path)
        results, errors = {}, {}
        if args.command == "demo":
            print("ДЕМО: ответы составлены вручную. Модель и сеть не используются.\n")
        elif not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("Задайте OPENAI_API_KEY или используйте команду demo.")
        for version in ("v1", "v2"):
            if args.command == "demo":
                predictions = read_json(ROOT / "examples" / f"{version}.json")
            else:
                predictions = {}
                for case in cases:
                    print(f"{version} / {case['id']}…", flush=True)
                    try:
                        predictions[case["id"]] = classify(case["text"], version, args.model)
                    except (ValueError, OSError) as error:
                        errors[f"{version}/{case['id']}"] = str(error)
                        predictions[case["id"]] = None
                    # Сохраняем прогресс после каждого запроса.
                    write_json(args.out / f"{version}.json", predictions)
            results[version] = evaluate(cases, predictions)
            show_summary(version, results[version])
        save_report(args.out, args.command, cases_path, results, getattr(args, "model", None), errors)
        return 1 if errors else 0
    except (ValueError, OSError) as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
