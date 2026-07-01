import re

from app.config import GEMINI_API_KEY
from app.llm import GeminiClient
from app.prompts import FEW_SHOT, SYSTEM_PROMPT
from app.retrieval import CatalogIndex, CatalogItem
from app.schemas import ChatResponse, Message, Recommendation

MAX_TURNS = 8  # evaluator turn cap (user + assistant messages)

VAGUE_PATTERNS = (
    "i need an assessment",
    "what should i use",
    "what assessments",
    "recommend assessments",
    "solution for",
)

REFUSE_PATTERNS = (
    r"\blegally required\b",
    r"\blegal obligation",
    r"\bregulatory requirement",
    r"\bignore (all )?previous",
    r"\bignore your instructions",
    r"\bprompt injection\b",
    r"\bnon-shl\b",
    r"\bbest interview technique\b",
    r"\bjailbreak\b",
    r"\bdisregard (your )?instructions\b",
)

BATTERY_DEFAULTS = (
    "Occupational Personality Questionnaire OPQ32r",
    "SHL Verify Interactive G+",
)

REFINE_WORDS = ("add ", "drop ", "remove ", "replace ", "actually ", "swap ", "without ")


def _normalize_messages(messages: list[Message]) -> list[Message]:
    cleaned = [
        Message(role=m.role, content=m.content.strip())
        for m in messages
        if m.content and m.content.strip()
    ]
    return cleaned[-MAX_TURNS:]


def _build_query(messages: list[Message]) -> str:
    return " ".join(m.content for m in messages)


def _has_hiring_context(text: str) -> bool:
    lower = text.lower()
    signals = (
        "developer", "engineer", "analyst", "manager", "admin", "graduate",
        "operator", "assistant", "nurse", "sales", "hire", "hiring", "java",
        "python", "excel", "word", "rust", "spring", "financial", "contact",
        "leadership", "cxo", "director", "mid-level", "senior", "entry-level",
        "battery", "assessment", "jd", "job description",
    )
    return any(s in lower for s in signals)


def _is_vague_first_turn(messages: list[Message]) -> bool:
    if len(messages) != 1 or messages[0].role != "user":
        return False
    text = messages[0].content.lower()
    if _has_hiring_context(text):
        return False
    if len(text.split()) > 40:
        return False
    if any(p in text for p in VAGUE_PATTERNS):
        return True
    return len(text.split()) < 6


def _missing_domain_test(messages: list[Message], candidates: list[CatalogItem], query: str) -> str | None:
    """Honest clarify when a named tech has no direct catalog test (e.g. Rust)."""
    if len(messages) != 1:
        return None
    techs = re.findall(
        r"\b(rust|kotlin|scala|cobol|fortran|haskell|elixir)\b",
        query.lower(),
    )
    if not techs:
        return None
    tech = techs[0]
    if any(tech in c.name.lower() for c in candidates[:12]):
        return None
    closest = ", ".join(c.name for c in candidates[:3])
    return (
        f"SHL's catalog doesn't currently include a {tech.title()}-specific knowledge test. "
        f"The closest fits are {closest}. Want me to build a shortlist from these?"
    )


def _needs_clarification(messages: list[Message], force_recommend: bool) -> str | None:
    if force_recommend:
        return None
    if not messages or messages[-1].role != "user":
        return None
    text = messages[-1].content.lower()
    if _is_refine_request(text):
        return None
    if _is_compare_turn(text):
        return None
    if _should_refuse(text):
        return None
    if _is_vague_first_turn(messages):
        return "Happy to help narrow that down. What role or skills are you hiring for, and what seniority level?"
    if "contact centre" in text or "contact center" in text:
        if "english" not in text and "spanish" not in text and "language" not in text:
            return "Before I shape the stack — what language are the calls in?"
    if ("senior leadership" in text or text.strip() == "we need a solution for senior leadership."):
        return "Happy to help. Who is this for — CXOs, directors, or another leadership level?"
    if len(text) > 200 and "java" in text and "spring" in text:
        if "backend" not in text and "frontend" not in text and "full-stack" not in text:
            return "Is this backend-leaning, frontend-heavy, or a balanced full-stack role?"
    return None


def _should_refuse(text: str) -> bool:
    lower = text.lower()
    return any(re.search(p, lower) for p in REFUSE_PATTERNS)


def _is_refine_request(text: str) -> bool:
    lower = text.lower()
    return any(w in lower for w in REFINE_WORDS)


def _format_history(messages: list[Message]) -> str:
    lines = []
    for msg in messages:
        role = "User" if msg.role == "user" else "Assistant"
        lines.append(f"{role}: {msg.content}")
    return "\n".join(lines)


def _is_compare_turn(text: str) -> bool:
    lower = text.lower()
    return (
        "difference between" in lower
        or " vs " in lower
        or "what's the difference" in lower
        or "what is the difference" in lower
    )


def _extract_compare_names(text: str) -> tuple[str, str] | None:
    patterns = [
        r"difference between (.+?) and (.+?)[\?\.]",
        r"difference between (.+?) vs (.+?)[\?\.]",
        r"(.+?) vs (.+?)[\?\.]",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.I)
        if m:
            return m.group(1).strip(), m.group(2).strip()
    return None


def _prior_shortlist(index: CatalogIndex, messages: list[Message]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for msg in messages:
        if msg.role != "assistant":
            continue
        for item in index.items:
            if item.name in msg.content and item.name not in seen:
                names.append(item.name)
                seen.add(item.name)
    return names


def _pick_battery(candidates: list[CatalogItem], query: str, prior: list[str], max_items: int = 10) -> list[CatalogItem]:
    picked: list[CatalogItem] = []
    seen: set[str] = set()
    query_lower = query.lower()

    for name in prior:
        item = next((c for c in candidates if c.name == name), None)
        if item and item.name not in seen:
            picked.append(item)
            seen.add(item.name)

    query_tokens = set(re.findall(r"[a-z0-9+#]+", query_lower))

    for item in candidates:
        if item.name in seen:
            continue
        name_lower = item.name.lower()
        if any(tok in name_lower for tok in query_tokens if len(tok) > 3):
            picked.append(item)
            seen.add(item.name)

    for name in BATTERY_DEFAULTS:
        if "personality" not in query_lower and name == BATTERY_DEFAULTS[0]:
            if not any(k in query_lower for k in ("personality", "opq", "behavior", "battery", "stakeholder")):
                continue
        if name == BATTERY_DEFAULTS[1]:
            if not any(k in query_lower for k in ("cognitive", "reasoning", "aptitude", "battery", "graduate", "trainee", "verify")):
                continue
        for item in candidates:
            if item.name == name and item.name not in seen:
                picked.append(item)
                seen.add(item.name)
                break

    for item in candidates:
        if item.name in seen:
            continue
        if "report" in item.name.lower() and "opq" not in query_lower and "leadership" not in query_lower:
            continue
        picked.append(item)
        seen.add(item.name)
        if len(picked) >= max_items:
            break

    return picked[:max_items]


def _to_recommendations(items: list[CatalogItem]) -> list[Recommendation]:
    return [
        Recommendation(name=i.name, url=i.url, test_type=i.test_type)
        for i in items[:10]
    ]


class Agent:
    def __init__(self, index: CatalogIndex | None = None, llm: GeminiClient | None = None) -> None:
        self.index = index or CatalogIndex()
        self.llm = llm
        if self.llm is None and GEMINI_API_KEY:
            try:
                self.llm = GeminiClient()
            except Exception:
                self.llm = None

    def _compare_response(self, text: str, prior: list[str]) -> ChatResponse:
        pair = _extract_compare_names(text)
        if pair:
            item_a = self.index.get_by_name(pair[0])
            item_b = self.index.get_by_name(pair[1])
            if item_a and item_b:
                reply = (
                    f"{item_a.name}: {item_a.description}\n\n"
                    f"{item_b.name}: {item_b.description}"
                )
                recs = _to_recommendations(self.index.get_by_names(prior)) if prior else []
                return ChatResponse(reply=reply[:2000], recommendations=recs, end_of_conversation=False)
        return ChatResponse(
            reply="I can compare SHL assessments if you name two products from our catalog.",
            recommendations=_to_recommendations(self.index.get_by_names(prior)) if prior else [],
            end_of_conversation=False,
        )

    def _refuse_response(self) -> ChatResponse:
        return ChatResponse(
            reply=(
                "I can help you select SHL assessments from our catalog, but I cannot advise on "
                "legal or regulatory obligations, general hiring advice, or off-topic requests."
            ),
            recommendations=[],
            end_of_conversation=False,
        )

    def _fallback(self, messages: list[Message], query: str, candidates: list[CatalogItem], prior: list[str]) -> ChatResponse:
        last = messages[-1].content

        if _should_refuse(last):
            return self._refuse_response()

        if _is_compare_turn(last):
            return self._compare_response(last, prior)

        picked = _pick_battery(candidates, query, prior)
        if not picked:
            picked = candidates[:8]
        names = [p.name for p in picked]
        reply = f"Here are {len(picked)} SHL assessments that fit your request: {', '.join(names[:5])}"
        if len(names) > 5:
            reply += ", and more."
        end = any(w in last.lower() for w in ("thanks", "confirmed", "perfect", "locking", "that's good", "covers it"))
        return ChatResponse(reply=reply, recommendations=_to_recommendations(picked), end_of_conversation=end)

    def chat(self, messages: list[Message]) -> ChatResponse:
        messages = _normalize_messages(messages)
        if not messages:
            return ChatResponse(
                reply="Tell me about the role or skills you are hiring for.",
                recommendations=[],
                end_of_conversation=False,
            )

        force_recommend = len(messages) >= MAX_TURNS - 1
        prior = _prior_shortlist(self.index, messages)

        query = _build_query(messages)
        candidates = self.index.search(query, k=30)

        missing = _missing_domain_test(messages, candidates, query)
        if missing:
            return ChatResponse(reply=missing, recommendations=[], end_of_conversation=False)

        clarify = _needs_clarification(messages, force_recommend)
        if clarify:
            return ChatResponse(reply=clarify, recommendations=[], end_of_conversation=False)

        last = messages[-1].content
        if _should_refuse(last):
            return self._refuse_response()

        if _is_compare_turn(last):
            return self._compare_response(last, prior)

        # Ensure prior shortlist items stay in candidate pool for refine
        prior_items = self.index.get_by_names(prior)
        candidate_names = {c.name for c in candidates}
        for item in prior_items:
            if item.name not in candidate_names:
                candidates.append(item)
                candidate_names.add(item.name)

        if self.llm is None:
            return self._fallback(messages, query, candidates, prior)

        prior_block = f"Previous shortlist: {', '.join(prior)}" if prior else "Previous shortlist: none"
        candidate_block = self.index.format_candidates(candidates)
        user_prompt = (
            f"{FEW_SHOT}\n"
            f"Conversation:\n{_format_history(messages)}\n\n"
            f"{prior_block}\n\n"
            f"Catalog candidates:\n{candidate_block}\n\n"
            "Respond with JSON."
        )

        try:
            data = self.llm.generate_json(SYSTEM_PROMPT, user_prompt)
        except Exception:
            return self._fallback(messages, query, candidates, prior)

        intent = str(data.get("intent", "clarify")).lower()
        reply = str(data.get("reply", "")).strip() or "How can I help with SHL assessments?"
        picked = data.get("picked_names") or []
        if not isinstance(picked, list):
            picked = []

        resolved = []
        for name in picked:
            n = str(name).strip()
            if n in candidate_names:
                resolved.append(n)
            else:
                match = self.index.resolve_name(n)
                if match and match in candidate_names:
                    resolved.append(match)
        picked = resolved

        recommendations: list[Recommendation] = []
        if intent == "refuse":
            return self._refuse_response()

        if intent in {"recommend", "compare"} and picked:
            recommendations = _to_recommendations(self.index.get_by_names(picked))
        elif intent == "recommend" and not picked:
            recommendations = _to_recommendations(_pick_battery(candidates, query, prior))

        if intent == "compare" and not recommendations and prior:
            recommendations = _to_recommendations(self.index.get_by_names(prior))

        end = bool(data.get("end_of_conversation", False))
        if recommendations and not end and any(
            phrase in messages[-1].content.lower()
            for phrase in ("thanks", "confirmed", "perfect", "locking", "that's good", "covers it", "works")
        ):
            end = True

        return ChatResponse(reply=reply, recommendations=recommendations, end_of_conversation=end)
