"""ContextEngine — главный компонент Styx.

На каждом turn'е собирает messages array, который Hermes отправит в LLM.
Контракт композиции — `.design/integrations/hermes-v1.md` § StyxContextEngine
и `.design/waves-v1.md` § «Поверхности».
"""
