"""One /ask response as a few readable lines, for the recorded demo.

    curl -s localhost:8077/ask -d '...' | python3 scripts/demo_brief.py LOG N

reads the response JSON on stdin and prints its outcome, the citation and the
stages the service timed for it, taken from the N-th /ask line of the
service's log at LOG. Nothing is computed here: every value printed is a
field of the response or of that log line. Standard library only, because
``scripts/demo.sh`` runs it with the system ``python3``.
"""

import json
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

WIDTH = 104
ASK_LINE = '{"method":"POST","path":"/ask"'
LOG_POLLS = 20
LOG_POLL_SECONDS = 0.1


def logged_asks(log: Path, wanted: int) -> list[dict[str, Any]]:
    """The /ask lines of the service log, waiting briefly until ``wanted`` are written.

    The service writes a request's line as the response leaves, so it can
    land a moment after the client has the body.
    """
    asks: list[dict[str, Any]] = []
    for _ in range(LOG_POLLS):
        lines = log.read_text("utf-8").splitlines()
        asks = [json.loads(line) for line in lines if line.startswith(ASK_LINE)]
        if len(asks) >= wanted:
            break
        time.sleep(LOG_POLL_SECONDS)
    return asks


def _wrapped(text: str, indent: str) -> str:
    return textwrap.fill(
        " ".join(text.split()), WIDTH, initial_indent=indent, subsequent_indent=indent
    )


def render(response: dict[str, Any], stages: list[str] | None) -> str:
    """The lines the demo shows for one response."""
    scope = response["source"]
    lines = [
        f"POST /ask  cik {scope['cik']}, fiscal_year {scope['fiscal_year']}",
        _wrapped(f"question: {response['question']}", "  "),
    ]
    if response["abstained"]:
        lines.append(f"abstained   {response['reason']}")
        lines.append(_wrapped(response["detail"], "  "))
    else:
        lines.append(f"answered    {response['text']}")
        for citation in response["citations"]:
            lines.append(
                f"  [{citation['marker']}] {citation['item']}, {citation['title']},"
                f" score {citation['score']}"
            )
            quote = textwrap.shorten(citation["quote"], 2 * WIDTH - 20, placeholder=" ...")
            lines.append(_wrapped(f'"{quote}"', "      "))
    if stages is not None:
        lines.append(f"  stages: {', '.join(stages)}")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    if len(argv) != 3 or not argv[2].isdigit() or int(argv[2]) < 1:
        print("usage: demo_brief.py LOG N  (the response JSON on stdin)", file=sys.stderr)
        return 2
    log, nth = Path(argv[1]), int(argv[2])
    response = json.loads(sys.stdin.read())
    if "abstained" not in response:
        print(f"not an answer: {response}", file=sys.stderr)
        return 1
    asks = logged_asks(log, nth)
    stages = list(asks[nth - 1]["stages"]) if len(asks) >= nth else None
    print(render(response, stages))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
