"""Причинное наблюдение завершённого когнитивного перехода.

Этот модуль не назначает агенту эмоциональную роль и не анализирует один
лишь тон собеседника.  Он получает завершённый turn целиком (стимул +
фактический ответ агента), извлекает проверяемое свидетельство о переходе и
возвращает ограниченную структуру для append-only журнала.

LLM здесь — наблюдатель/экстрактор. Авторитетным состоянием остаётся
детерминированный reducer в :mod:`styx.emotional.state`; текстовые ярлыки не
попадают в активный контекст.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Literal

from styx.emotional.state import (
    EMOTIONAL_AXIS_MAX,
    EMOTIONAL_AXIS_MIN,
    EmotionalVector,
)
from styx.llm import (
    LLMRateLimiter,
    OllamaChatClient,
    OllamaTerminalError,
    OllamaTransientError,
)

log = logging.getLogger(__name__)

TURN_OBSERVATION_TIMEOUT_S = 8.0
TURN_OBSERVATION_MAX_ATTEMPTS = 1
REACTION_GAIN = 0.55
MAX_CAUSE_CHARS = 280
MAX_TURN_TEXT_CHARS = 12_000
MAX_REFERENCED_CAUSES = 8
MAX_PRIOR_CAUSES = 8

CauseStatus = Literal["unknown", "active", "resolved", "superseded"]
CauseClass = Literal[
    "semantic_alignment",
    "task_uncertainty",
    "constraint_pressure",
    "execution_risk",
    "conflicting_signals",
    "goal_progress",
    "discovery",
    "interpersonal_tension",
    "resolution",
    "unknown",
]
CauseSubject = Literal[
    "response_correctness",
    "task_completion",
    "constraint_compliance",
    "tool_outcome",
    "relationship_alignment",
    "uncertainty_resolution",
    "external_event",
    "unknown",
]
AttentionPolicy = Literal[
    "preserve_direction",
    "verify_correspondence",
    "surface_ambiguity",
    "explore_connections",
]
VerificationDepth = Literal["normal", "high"]
BranchBudget = Literal["narrow", "balanced", "broad"]
ClosurePolicy = Literal["normal", "resist_premature_closure"]

_CAUSE_STATUSES = {"unknown", "active", "resolved", "superseded"}
_CAUSE_CLASSES = {
    "semantic_alignment",
    "task_uncertainty",
    "constraint_pressure",
    "execution_risk",
    "conflicting_signals",
    "goal_progress",
    "discovery",
    "interpersonal_tension",
    "resolution",
    "unknown",
}
_CAUSE_SUBJECTS = {
    "response_correctness", "task_completion", "constraint_compliance",
    "tool_outcome", "relationship_alignment", "uncertainty_resolution",
    "external_event", "unknown",
}
_ATTENTION_POLICIES = {
    "preserve_direction",
    "verify_correspondence",
    "surface_ambiguity",
    "explore_connections",
}
_VERIFICATION_DEPTHS = {"normal", "high"}
_BRANCH_BUDGETS = {"narrow", "balanced", "broad"}
_CLOSURE_POLICIES = {"normal", "resist_premature_closure"}

_EMAIL_RE = re.compile(
    r"(?i)(?<![\w.+-])[\w.+-]+@[a-z0-9-]+(?:\.[a-z0-9-]+)+(?![\w.-])"
)
_BEARER_RE = re.compile(r"(?i)\bbearer\s+[a-z0-9._~+/=-]{8,}")
_NAMED_SECRET_RE = re.compile(
    r"(?i)\b(api[\s_-]*key|access[\s_-]*token|token|secret|password)"
    r"(\s*[:=]\s*)(?:[\"']?)[^\s,;\"']{6,}(?:[\"']?)"
)
_KEY_SHAPE_RE = re.compile(r"(?<![\w-])(?:sk|pk|rk)-[a-zA-Z0-9_-]{12,}")
_PHONE_CANDIDATE_RE = re.compile(r"(?<!\w)(?:\+?\d[\s().-]*){10,15}(?!\w)")


@dataclass(frozen=True)
class CognitivePosture:
    """Не-стилевая проекция состояния на проверяемые решения."""

    attention: AttentionPolicy = "preserve_direction"
    verification_depth: VerificationDepth = "normal"
    branch_budget: BranchBudget = "balanced"
    closure_policy: ClosurePolicy = "normal"

    def as_dict(self) -> dict[str, str]:
        return {
            "attention": self.attention,
            "verification_depth": self.verification_depth,
            "branch_budget": self.branch_budget,
            "closure_policy": self.closure_policy,
        }


@dataclass(frozen=True)
class AffectiveAssessment:
    """Свидетельство о завершённом turn, а не готовая эмоция."""

    stimulus: EmotionalVector
    reaction: EmotionalVector
    cause_class: CauseClass
    cause_summary: str
    intensity: float
    confidence: float
    cause_status: CauseStatus
    posture: CognitivePosture
    cause_subject: CauseSubject = "unknown"
    updates_event_ids: tuple[int, ...] = ()
    reaffirms_event_ids: tuple[int, ...] = ()
    revises_event_ids: tuple[int, ...] = ()

    @property
    def weighted_delta(self) -> EmotionalVector:
        """Вклад в residue с учётом силы и неопределённости."""
        gain = REACTION_GAIN * self.intensity * self.confidence
        return EmotionalVector(
            valence=self.reaction.valence * gain,
            arousal=self.reaction.arousal * gain,
            dominance=self.reaction.dominance * gain,
        )


SYSTEM_PROMPT = """\
Ты фиксируешь причинный осадок уже завершённого когнитивного акта агента.
Это НЕ задача назначить настроение, подобрать тон или описать стиль ответа.

Различай:
- stimulus_vad: эмоциональные координаты входящей реплики/обстоятельства;
- reaction_vad: изменение внутреннего фона агента, которое подтверждается тем,
  что агент заметил, проверил, выбрал и удержал в своём фактическом ответе;
- cause_class: один контролируемый класс причины из JSON-схемы ниже;
- cause_summary: конкретная причина реакции, без имён людей и приватных данных;
- cause_subject: контролируемая область причины, позволяющая различать причины
  одного класса без передачи свободной прозы обратно наблюдателю;
- confidence: насколько вывод подтверждается causal line turn'а;
- intensity: сила именно этой причины;
- cause_status: active если причина продолжает действовать после ответа,
  resolved если исчерпана, superseded если перебита новым обстоятельством,
  unknown если данных недостаточно.
- updates_event_ids: evidence_id активных прежних причин, которые этот
  ход разрешает/вытесняет. Не угадывай id, которых нет в prior context;
- reaffirms_event_ids: evidence_id активных прежних причин, которые ход лишь
  подтверждает. Это продлевает их инерцию, но не добавляет ту же delta повторно.
- revises_event_ids: ровно один evidence_id продолжающейся причины, силу и
  cognitive_posture которой новое свидетельство уточняет. Reducer применит
  только разницу нового и прежнего support.

Не зеркаль тон пользователя автоматически. Одинаковый stimulus может дать
разную reaction в зависимости от цели и фактического выбора агента. Если
ответ не подтверждает реакцию, ставь малую confidence и близкую к нулю
reaction_vad.

cognitive_posture описывает только когнитивные решения следующего хода, не
манеру речи:
- attention: preserve_direction | verify_correspondence | surface_ambiguity |
  explore_connections
- verification_depth: normal | high
- branch_budget: narrow | balanced | broad
- closure_policy: normal | resist_premature_closure

Верни только JSON:
{
  "stimulus_vad": {"valence": number, "arousal": number, "dominance": number},
    "reaction_vad": {"valence": number, "arousal": number, "dominance": number},
  "cause_class": "semantic_alignment|task_uncertainty|constraint_pressure|execution_risk|conflicting_signals|goal_progress|discovery|interpersonal_tension|resolution|unknown",
  "cause_subject": "response_correctness|task_completion|constraint_compliance|tool_outcome|relationship_alignment|uncertainty_resolution|external_event|unknown",
  "cause_summary": string,
  "intensity": number,
  "confidence": number,
  "cause_status": "unknown|active|resolved|superseded",
  "updates_event_ids": [positive_integer],
  "reaffirms_event_ids": [positive_integer],
  "revises_event_ids": [positive_integer],
  "cognitive_posture": {
    "attention": string,
    "verification_depth": string,
    "branch_budget": string,
    "closure_policy": string
  }
}

Все VAD в [-1,+1], intensity/confidence в [0,1]."""


def redact_cause_summary(value: str) -> str:
    """Детерминированно удалить частые секреты/PII до persistence.

    Это последний защитный барьер, а не попытка распознать все возможные
    персональные данные. Свободная проза остаётся только audit evidence и
    никогда не должна возвращаться в active prompt.
    """

    redacted = _EMAIL_RE.sub("[redacted-email]", value)
    redacted = _BEARER_RE.sub("Bearer [redacted-secret]", redacted)
    redacted = _KEY_SHAPE_RE.sub("[redacted-secret]", redacted)
    redacted = _NAMED_SECRET_RE.sub(
        lambda match: f"{match.group(1)}{match.group(2)}[redacted-secret]",
        redacted,
    )

    def _redact_phone(match: re.Match[str]) -> str:
        candidate = match.group(0)
        digits = sum(character.isdigit() for character in candidate)
        return "[redacted-phone]" if 10 <= digits <= 15 else candidate

    return _PHONE_CANDIDATE_RE.sub(_redact_phone, redacted)


def canonical_turn_hash(
    *, session_id: str | None, user_message: str, assistant_response: str
) -> str:
    """Стабильный безопасный fingerprint без сохранения сырого текста в ключе."""
    payload = json.dumps(
        [session_id or "", user_message, assistant_response],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _finite_number(raw: object, *, name: str, lo: float, hi: float) -> float:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(f"{name} должен быть number")
    value = float(raw)
    if not math.isfinite(value) or not lo <= value <= hi:
        raise ValueError(f"{name} вне [{lo}, {hi}]")
    return value


def _vad(raw: object, *, name: str) -> EmotionalVector:
    if not isinstance(raw, dict):
        raise ValueError(f"{name} должен быть object")
    return EmotionalVector(
        valence=_finite_number(
            raw.get("valence"), name=f"{name}.valence",
            lo=EMOTIONAL_AXIS_MIN, hi=EMOTIONAL_AXIS_MAX,
        ),
        arousal=_finite_number(
            raw.get("arousal"), name=f"{name}.arousal",
            lo=EMOTIONAL_AXIS_MIN, hi=EMOTIONAL_AXIS_MAX,
        ),
        dominance=_finite_number(
            raw.get("dominance"), name=f"{name}.dominance",
            lo=EMOTIONAL_AXIS_MIN, hi=EMOTIONAL_AXIS_MAX,
        ),
    )


def validate_assessment(raw: object) -> AffectiveAssessment:
    if not isinstance(raw, dict):
        raise ValueError("assessment должен быть object")

    cause = raw.get("cause_summary")
    if not isinstance(cause, str) or not cause.strip():
        raise ValueError("cause_summary должен быть непустой строкой")
    cause = redact_cause_summary(" ".join(cause.split()))[:MAX_CAUSE_CHARS]

    cause_class = raw.get("cause_class")
    if cause_class not in _CAUSE_CLASSES:
        raise ValueError("неизвестный cause_class")
    cause_subject = raw.get("cause_subject", "unknown")
    if cause_subject not in _CAUSE_SUBJECTS:
        raise ValueError("неизвестный cause_subject")

    status = raw.get("cause_status")
    if status not in _CAUSE_STATUSES:
        raise ValueError("неизвестный cause_status")

    posture_raw = raw.get("cognitive_posture")
    if not isinstance(posture_raw, dict):
        raise ValueError("cognitive_posture должен быть object")
    attention = posture_raw.get("attention")
    verification = posture_raw.get("verification_depth")
    branch = posture_raw.get("branch_budget")
    closure = posture_raw.get("closure_policy")
    if attention not in _ATTENTION_POLICIES:
        raise ValueError("неизвестный cognitive_posture.attention")
    if verification not in _VERIFICATION_DEPTHS:
        raise ValueError("неизвестный cognitive_posture.verification_depth")
    if branch not in _BRANCH_BUDGETS:
        raise ValueError("неизвестный cognitive_posture.branch_budget")
    if closure not in _CLOSURE_POLICIES:
        raise ValueError("неизвестный cognitive_posture.closure_policy")

    def _event_ids(field_name: str) -> tuple[int, ...]:
        values = raw.get(field_name, [])
        if not isinstance(values, list) or len(values) > MAX_REFERENCED_CAUSES:
            raise ValueError(f"{field_name} должен быть bounded array")
        unique: list[int] = []
        for item in values:
            if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
                raise ValueError(f"{field_name} содержит невалидный id")
            if item not in unique:
                unique.append(item)
        return tuple(unique)

    updates = _event_ids("updates_event_ids")
    reaffirms = _event_ids("reaffirms_event_ids")
    revises = _event_ids("revises_event_ids")
    if updates and status not in {"resolved", "superseded"}:
        raise ValueError(
            "updates_event_ids допустим только для resolved/superseded"
        )
    if reaffirms and (status != "active" or updates or revises):
        raise ValueError(
            "reaffirms_event_ids допустим только для чистого active reaffirmation"
        )
    if revises and (status != "active" or updates or reaffirms or len(revises) != 1):
        raise ValueError("revises_event_ids требует ровно одну active причину")
    if (set(updates) & set(reaffirms)) or (set(updates) & set(revises)) or (set(reaffirms) & set(revises)):
        raise ValueError("lifecycle event id arrays не должны пересекаться")

    return AffectiveAssessment(
        stimulus=_vad(raw.get("stimulus_vad"), name="stimulus_vad"),
        reaction=_vad(raw.get("reaction_vad"), name="reaction_vad"),
        cause_class=cause_class,
        cause_summary=cause,
        intensity=_finite_number(
            raw.get("intensity"), name="intensity", lo=0.0, hi=1.0
        ),
        confidence=_finite_number(
            raw.get("confidence"), name="confidence", lo=0.0, hi=1.0
        ),
        cause_status=status,
        posture=CognitivePosture(
            attention=attention,
            verification_depth=verification,
            branch_budget=branch,
            closure_policy=closure,
        ),
        cause_subject=cause_subject,
        updates_event_ids=updates,
        reaffirms_event_ids=reaffirms,
        revises_event_ids=revises,
    )


def _bounded_turn_prompt(
    *,
    user_message: str,
    assistant_response: str,
    prior_context: dict[str, Any] | None,
    conversation_history: list[dict[str, Any]] | None,
    tool_events: list[dict[str, Any]] | None,
) -> str:
    history: list[dict[str, str]] = []
    for item in (conversation_history or [])[-6:]:
        role = item.get("role")
        content = item.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str):
            continue
        history.append({"role": role, "content": content[:2_000]})
    safe_prior: dict[str, Any] = {}
    if isinstance(prior_context, dict):
        safe_prior = {
            key: prior_context.get(key)
            for key in ("state_id", "observed_at", "coordinates", "confidence")
            if key in prior_context
        }
        safe_causes: list[dict[str, Any]] = []
        raw_causes = prior_context.get("causes")
        if isinstance(raw_causes, list):
            for item in raw_causes[:MAX_PRIOR_CAUSES]:
                if not isinstance(item, dict):
                    continue
                # Deliberately exclude cause/cause_summary and arbitrary
                # metadata: free-form persisted evidence is never prompt input.
                evidence_id = item.get("evidence_id")
                if (
                    isinstance(evidence_id, bool)
                    or not isinstance(evidence_id, int)
                    or evidence_id <= 0
                ):
                    continue
                cause_class = item.get("cause_class")
                status_value = item.get("status")
                posture = item.get("cognitive_posture")
                safe_posture = None
                if isinstance(posture, dict) and all(
                    posture.get(field) in allowed
                    for field, allowed in (
                        ("attention", _ATTENTION_POLICIES),
                        ("verification_depth", _VERIFICATION_DEPTHS),
                        ("branch_budget", _BRANCH_BUDGETS),
                        ("closure_policy", _CLOSURE_POLICIES),
                    )
                ):
                    safe_posture = {
                        field: posture[field]
                        for field in (
                            "attention", "verification_depth",
                            "branch_budget", "closure_policy",
                        )
                    }
                safe_causes.append({
                    "evidence_id": evidence_id,
                    "cause_class": (
                        cause_class if cause_class in _CAUSE_CLASSES else "unknown"
                    ),
                    "cause_subject": item.get("cause_subject")
                    if item.get("cause_subject") in _CAUSE_SUBJECTS else "unknown",
                    "status": (
                        status_value if status_value in _CAUSE_STATUSES else "unknown"
                    ),
                    "cause_active": item.get("cause_active") is True,
                    "intensity": item.get("intensity")
                    if isinstance(item.get("intensity"), (int, float))
                    and not isinstance(item.get("intensity"), bool) else None,
                    "confidence": item.get("confidence")
                    if isinstance(item.get("confidence"), (int, float))
                    and not isinstance(item.get("confidence"), bool) else None,
                    "observed_at": item.get("observed_at")
                    if isinstance(item.get("observed_at"), str) else None,
                    "lease_expires_at": item.get("lease_expires_at")
                    if isinstance(item.get("lease_expires_at"), str) else None,
                    "cognitive_posture": safe_posture,
                })
        safe_prior["causes"] = safe_causes
        safe_prior["omitted_cause_count"] = max(
            0, len(raw_causes) - len(safe_causes)
        ) if isinstance(raw_causes, list) else 0
    safe_tools: list[dict[str, str]] = []
    for item in (tool_events or [])[-8:]:
        if not isinstance(item, dict):
            continue
        safe: dict[str, str] = {}
        for key in ("name", "tool", "status", "result", "error"):
            value = item.get(key)
            if isinstance(value, (str, int, float, bool)):
                safe[key] = str(value)[:2_000]
        if safe:
            safe_tools.append(safe)
    payload = {
        "prior_causal_context": safe_prior,
        "recent_dialogue_context": history,
        "user_message": user_message[:MAX_TURN_TEXT_CHARS],
        "assistant_response": assistant_response[:MAX_TURN_TEXT_CHARS],
        "tool_events": safe_tools,
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


@dataclass
class TransitionMetrics:
    calls: int = 0
    applied: int = 0
    failures: int = 0
    duplicates: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def bump(self, name: str) -> None:
        with self._lock:
            setattr(self, name, int(getattr(self, name)) + 1)

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "calls": self.calls,
                "applied": self.applied,
                "failures": self.failures,
                "duplicates": self.duplicates,
            }


class TransitionObserver:
    """Fail-open extractor причинного осадка completed turn."""

    def __init__(
        self,
        llm: OllamaChatClient,
        rate_limit: LLMRateLimiter,
        *,
        timeout_s: float = TURN_OBSERVATION_TIMEOUT_S,
    ) -> None:
        self._llm = llm
        self._rate = rate_limit
        self._timeout_s = timeout_s
        self.metrics = TransitionMetrics()

    def observe(
        self,
        *,
        user_message: str,
        assistant_response: str,
        prior_context: dict[str, Any] | None = None,
        conversation_history: list[dict[str, Any]] | None = None,
        tool_events: list[dict[str, Any]] | None = None,
    ) -> AffectiveAssessment | None:
        self.metrics.bump("calls")
        if not user_message.strip() or not assistant_response.strip():
            self.metrics.bump("failures")
            return None
        if not self._rate.try_acquire():
            self.metrics.bump("failures")
            return None
        try:
            raw = self._llm.chat_json(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": _bounded_turn_prompt(
                            user_message=user_message,
                            assistant_response=assistant_response,
                            prior_context=prior_context,
                            conversation_history=conversation_history,
                            tool_events=tool_events,
                        ),
                    },
                ],
                timeout_s=self._timeout_s,
                max_attempts=TURN_OBSERVATION_MAX_ATTEMPTS,
            )
            return validate_assessment(raw)
        except (OllamaTransientError, OllamaTerminalError, ValueError) as exc:
            log.debug("affective transition observation skipped: %s", exc)
        except Exception as exc:  # noqa: BLE001 - fail-open turn path
            log.warning("affective transition observation failed: %s", exc)
        self.metrics.bump("failures")
        return None


def make_transition_observer(
    *,
    base_url: str,
    model: str,
    timeout_s: float = TURN_OBSERVATION_TIMEOUT_S,
    capacity: int = 2,
    refill_per_second: float = 1.0,
) -> TransitionObserver:
    return TransitionObserver(
        llm=OllamaChatClient(
            base_url=base_url,
            model=model,
            timeout_s=timeout_s,
            max_attempts=TURN_OBSERVATION_MAX_ATTEMPTS,
        ),
        rate_limit=LLMRateLimiter(
            capacity=capacity, refill_per_second=refill_per_second
        ),
        timeout_s=timeout_s,
    )
