"""Channels для pre_llm_inject framework (волна 15).

Каждый channel — pure function ``(handle, hermes_kwargs) → str | None``.
Регистрируются в provider initialize() через ``pre_llm_inject.configure``.
"""
