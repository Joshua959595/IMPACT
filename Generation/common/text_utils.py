"""Framework-agnostic text/JSON parsing helpers."""
import re
from typing import List, Optional


def render(template: str, **kwargs) -> str:
    for k in list(kwargs.keys()):
        template = template.replace("{ " + k + " }", "{" + k + "}")
    try:
        return template.format(**kwargs).strip()
    except KeyError as e:
        raise KeyError(f"Missing template variable: {e.args[0]}") from e


def extract_first_tag(text: str, tag: str) -> Optional[str]:
    open_tag = f"<{tag}>"
    close_tag = f"</{tag}>"
    if open_tag not in text or close_tag not in text:
        return None
    return text.split(open_tag, 1)[1].split(close_tag, 1)[0].strip()


def extract_all_tagged(text: str, tag: str) -> List[str]:
    open_tag = f"<{tag}>"
    close_tag = f"</{tag}>"
    out = []
    s = text
    while open_tag in s and close_tag in s:
        chunk = s.split(open_tag, 1)[1].split(close_tag, 1)[0].strip()
        out.append(chunk)
        s = s.split(close_tag, 1)[1]
    return out


def extract_json_object(text: str) -> str:
    if not text:
        return ""
    t = text.strip()
    t = re.sub(r"^```json\s*", "", t, flags=re.IGNORECASE).strip()
    t = re.sub(r"^```\s*", "", t).strip()
    t = re.sub(r"\s*```$", "", t).strip()
    m = re.search(r"\{.*\}", t, flags=re.DOTALL)
    return m.group(0).strip() if m else t
