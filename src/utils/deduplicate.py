from typing import Dict, List


def deduplicate_by_text(docs: List[Dict]) -> List[Dict]:
    seen: set[str] = set()
    result: List[Dict] = []
    for doc in docs:
        text = doc.get("text", "")
        if text not in seen:
            seen.add(text)
            result.append(doc)
    return result
