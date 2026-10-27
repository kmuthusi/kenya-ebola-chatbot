"""Check that generated answers cite retrieved PDF pages."""

import re

from bounded_composer import REFUSAL


CITATION_RE = re.compile(r"\bPDF Page\s+(\d+)\b", re.IGNORECASE)
END_CITATION_RE = re.compile(r"\(PDF Page\s+\d+\)[.!?]*$", re.IGNORECASE)


def verify_answer(answer: str, evidence: list[dict]) -> dict:
    if answer == REFUSAL:
        return {"verified_answer": answer, "issues": []}
    cited = {int(match) for match in CITATION_RE.findall(answer)}
    allowed = {item["page"] for item in evidence}
    issues = []
    if not cited:
        issues.append("The answer has no PDF page citation.")
    if cited - allowed:
        issues.append(f"Citations outside retrieved evidence: {sorted(cited - allowed)}")
    if any(
        line.lstrip().startswith(("- ", "* ", "• "))
        and not END_CITATION_RE.search(line.rstrip())
        for line in answer.splitlines()
    ):
        issues.append("Every answer bullet needs a PDF page citation.")
    return {
        "verified_answer": answer if not issues else REFUSAL,
        "issues": issues,
    }
