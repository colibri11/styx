"""Background workers Styx — отдельный процесс ``styx-worker``.

Здесь живёт LLM drain-loop (волна 7a), lifecycle sweep (7b), recall
classifier handler (7c) и emotional baseline tick (7d). Hot-path
sentiment живёт в ``styx.emotional`` — он inline в Hermes-процессе, а
не в worker'е.
"""
