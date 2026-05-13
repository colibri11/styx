"""Эмоциональная проекция агента — VAD-оси.

Hot-path sentiment (волна 7d) — inline в ``sync_turn``: extract_vad на
peer-реплике (user_content), append delta к ``emotional_state``.

Slow-path baseline aggregator — periodic task в worker-runtime: EMA
α=0.98 над окном 60 мин, UPSERT в ``emotional_baseline``.

Recall пробрасывает baseline в composite scoring как
``emotional_resonance`` фактор.
"""
