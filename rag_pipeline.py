"""
rag_pipeline.py — rerank and relevance release.
"""

import re
from typing import List, Dict, Any


def is_overview_question(question: str) -> bool:
    """Recognize requests about the document itself, rather than patient care."""
    normalized = re.sub(r"[^a-z0-9 ]", " ", question.lower())
    words = set(normalized.split())
    clinical_terms = {
        "isolation", "treatment", "dose", "dosing", "patient", "patients",
        "contact", "contacts", "testing", "laboratory", "symptom", "symptoms",
        "ppe", "discharge", "quarantine", "burial", "pregnancy", "triage",
        "diagnosis", "diagnostic", "fluid", "fluids", "shock", "referral",
    }
    if words & clinical_terms:
        return False
    return bool(words & {"guideline", "guidelines", "document", "pdf"}) and bool(
        words & {"about", "overview", "summarize", "summary", "describe",
                 "cover", "covers", "scope", "topics", "contain", "contains"}
    )


def overview_evidence(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Use the publication's own scope, foreword and purpose for an overview."""
    excerpts = []
    sections = (
        ("These guidelines provide", None),
        ("These Case Management Guidelines", "The Constitution"),
        ("This guideline focuses specifically", "Ebola virus disease"),
        ("1.1 Purpose", "1.2 Objectives"),
    )
    for start, end in sections:
        for page in pages:
            content = page["text"]
            offset = content.find(start)
            if offset < 0:
                continue
            if start == "1.1 Purpose" and "To provide" not in content[offset:offset + 100]:
                continue
            finish = content.find(end, offset) if end else -1
            excerpt = content[offset:finish if finish >= 0 else None].strip()
            if excerpt:
                excerpts.append({"page": page["page"], "text": excerpt})
            break
    return excerpts if len(excerpts) >= 2 else []


def is_general_management_question(question: str) -> bool:
    """Recognize broad care requests that need the management chapter."""
    words = set(re.sub(r"[^a-z0-9 ]", " ", question.lower()).split())
    specific_topics = {
        "shock", "pregnancy", "pregnant", "child", "children", "dose", "dosing",
        "fluid", "fluids", "isolation", "testing", "diagnosis", "malaria",
        "symptom", "symptoms", "bleeding", "contact", "contacts", "discharge",
        "therapeutic", "therapeutics", "vaccine", "vaccination", "ppe",
    }
    return (
        bool(words & {"manage", "management", "care", "treat"})
        and bool(words & {"ebola", "filovirus", "patient", "patients"})
        and not bool(words & specific_topics)
        and not is_overview_question(question)
    )


def management_evidence(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Select the actual care chapter instead of references and contents pages."""
    sections = (
        ("3. Management of Suspected and Confirmed Filovirus Disease Cases", 2200),
        ("2.4 Isolation", 1000),
        ("Fluid replacement", 1900),
    )
    evidence = []
    for heading, limit in sections:
        for page in pages:
            offset = page["text"].find(heading)
            if offset < 0 or (heading != "Fluid replacement" and offset > 120):
                continue
            if heading == "Fluid replacement" and "Nutrition" not in page["text"]:
                continue
            evidence.append({"page": page["page"], "text": page["text"][offset:offset + limit].strip()})
            break
    return evidence if len(evidence) >= 2 else []


def exclude_bibliography(question: str, candidates: List[Dict[str, Any]],
                         pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep reference-list chunks out of ordinary guideline answers."""
    if re.search(r"\b(references?|bibliography|citations?)\b", question, re.I):
        return candidates
    first_reference_page = next(
        (
            page["page"]
            for page in pages
            if re.search(r"(?m)^\s*\d+\.\s+References\s*$", page["text"][:250])
        ),
        None,
    )
    if first_reference_page is None:
        return candidates
    filtered = [item for item in candidates if item.get("page", 0) < first_reference_page]
    return filtered or candidates

def _simple_relevance_score(candidate: Dict[str, Any]) -> float:
    """Combine embedding similarity (already inverted in vector_store) + page bonus."""
    s = candidate.get("score", 0.0) or 0.0
    page_bonus = 0.05 if candidate.get("page") else 0.0
    return s + page_bonus

def rerank_and_release(question: str, candidates: List[Dict[str, Any]], intent: str,
                       limit: int = 5) -> Dict[str, Any]:
    # Basic reranking
    ranked = sorted(candidates, key=_simple_relevance_score, reverse=True)
    evidence = ranked[:len(ranked)]

    # Refusal criteria: no evidence or extremely low scores
    if not evidence or all((c.get("score") or 0) < 0.2 for c in evidence):
        return {"evidence": evidence[:2], "release": "refusal"}

    return {"evidence": evidence[:limit], "release": "grounded"}
