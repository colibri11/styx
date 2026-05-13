"""Helpers для multi-modal content shape (мини-волна 26.8 round 6).

Pi-embedded-runner (OpenClaw 2026.5.7) передаёт messages в формате
multi-part content:

    {"role": "user", "content": [{"type": "text", "text": "..."}, ...]}

вместо классического

    {"role": "user", "content": "..."}

До этой мини-волны Styx-core hot path (`_find_last_user_text`,
`_sanitize_styx_blocks`) обрабатывал только string-content и
pass-through'ил multi-modal (с TODO волны 27+). В production это
приводило к тому что `build_salient_block` не находил ни одного
user-message → возвращал None → `system_prompt_addition` был пуст →
hook возвращал undefined → boевые агенты не видели salient.

Этот module даёт unified extractor для text-частей multi-modal
content. Image / audio / другие non-text parts пропускаются — Styx
работает на тексте (embedding'и считаются над plain text).

Концептуально это не extension семантики Styx — это адаптация под
канал доставки content'а. Pi-embedded shape vs legacy shape должны
давать одинаковое поведение salient/recall.
"""

from __future__ import annotations

from typing import Any


def extract_text_from_content(content: Any) -> str | None:
    """Унифицированно извлекает plain text из content любого
    допустимого shape.

    Поддерживает три формы:
    - ``str`` — возвращается как есть (после strip-check'а на пустоту).
    - ``list[dict]`` (multi-modal pi-embedded shape) — конкатенирует
      text-части через ``"\\n"``. Non-text parts (image / audio /
      tool_use / etc.) пропускаются.
    - ``None`` / иные типы — ``None``.

    Возвращает stripped non-empty string или None если контента нет.

    Безопасно для unknown / future content shapes — gracefully
    игнорирует unrecognized part types.
    """
    if isinstance(content, str):
        text = content.strip()
        return text if text else None

    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            # Anthropic / OpenAI Responses API style: {type: "text", text: "..."}.
            if part.get("type") == "text":
                txt = part.get("text")
                if isinstance(txt, str) and txt.strip():
                    parts.append(txt)
            # Альтернативный shape некоторых SDK: {type: "input_text", text: "..."}.
            elif part.get("type") == "input_text":
                txt = part.get("text")
                if isinstance(txt, str) and txt.strip():
                    parts.append(txt)
        if parts:
            return "\n".join(parts).strip() or None

    return None


def replace_text_in_content(content: Any, replacer: "callable[[str], str]") -> Any:
    """Применяет ``replacer(text) -> text`` к text-частям content'а.

    Используется `_sanitize_styx_blocks` для вырезания family
    `<styx-*>...</styx-*>` и legacy `<styx>...</styx>` маркеров из
    multi-modal content. Возвращает новый content того же shape (str
    или list of parts) с обновлёнными text-частями. Non-text parts
    остаются как есть.

    Если после replacement весь text-content становится пустой —
    возвращает None (caller интерпретирует как «message целиком
    удалить»).
    """
    if isinstance(content, str):
        new_text = replacer(content)
        if not new_text.strip():
            return None
        return new_text

    if isinstance(content, list):
        out_parts: list[Any] = []
        has_text = False
        for part in content:
            if not isinstance(part, dict):
                out_parts.append(part)
                continue
            ptype = part.get("type")
            if ptype in ("text", "input_text"):
                txt = part.get("text")
                if isinstance(txt, str):
                    new_txt = replacer(txt)
                    if new_txt.strip():
                        new_part = dict(part)
                        new_part["text"] = new_txt
                        out_parts.append(new_part)
                        has_text = True
                    # Если text-part стал empty — drop'аем его, но не
                    # удаляем целиком message (могут быть другие parts
                    # типа image).
                else:
                    out_parts.append(part)
            else:
                out_parts.append(part)
        if not out_parts:
            return None
        # Если ни одного text-part'а не осталось и больше parts'ов
        # тоже нет (out_parts может содержать только non-text после
        # удаления text'ов) — caller проверяет по результату.
        if not has_text and all(
            isinstance(p, dict)
            and p.get("type") not in ("text", "input_text")
            for p in out_parts
        ):
            # Все text-parts были sanitize'нуты до пустоты. Но
            # остались non-text parts (image и т.п.). Сохраняем
            # message — он содержит non-text content (но без текста).
            return out_parts
        return out_parts

    # Unknown shape — pass-through (или None? Pass-through безопаснее).
    return content
