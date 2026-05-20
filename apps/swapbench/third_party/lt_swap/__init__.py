"""LT-Swap (Long-Tail Swap) benchmark generator.

Shipped unchanged from https://github.com/facebookresearch/lt-swap at the
upstream commit SHA pinned in ``VERSION``. Per-file copyright headers are
intact. The upstream multi-prompter (``mp_main.py``, ``mp_utils.py``) is
not invoked from EgoBabyVLM; instead, the Hydra wrappers in
``apps.swapbench.longtail_swap`` use the async worker pool in
``apps.swapbench.utils.llm_runner``, which targets an OpenAI-compatible
endpoint (e.g. a local vLLM server).
"""
