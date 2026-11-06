"""Conservative byte-based context estimate; provider tokenizers may differ."""
import os
from src import config


def completion_budget(messages, requested):
    context = int(os.environ.get('LLM_CONTEXT_LEN', '32768'))
    # UTF-8 byte count is deliberately conservative, with message/template reserve.
    prompt_estimate = sum(len(str(m.get('content', '')).encode('utf-8')) + 64 for m in messages)
    room = context - prompt_estimate - 512
    cap = min(int(requested), config.LLM_MAX_TOKEN_BUDGET, room)
    if cap <= 0:
        raise ValueError('No positive generation budget remains; shorten input or configure context')
    return cap
