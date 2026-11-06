"""Shared single-model execution, validated outputs and bound checkpoints."""
import json
import os
from pathlib import Path

from src import config
from src.llm.client import LLMError, extract_json, generate, resolve
from src.llm.parallel import run_concurrent
from src.llm.budget import completion_budget
from src.tasks.b3_provenance import digest, source_digest

SUMMARY_FIELDS = ('diagnoses', 'surgeries', 'medications',
                  'key_abnormal_labs', 'key_imaging_findings')
RUNNER_VERSION = 'single-model-v2'


def binary_prediction(parsed, events):
    pred = parsed.get('predictions', parsed) if isinstance(parsed, dict) else None
    if not isinstance(pred, dict) or any(e not in pred for e in events):
        raise ValueError('Prediction must include every event key')
    result = {}
    for event in events:
        value = pred[event]
        if type(value) is int and value in (0, 1):
            result[event] = value
        elif type(value) is str and value in ('0', '1'):
            result[event] = int(value)
        else:
            raise ValueError(f'Invalid binary value for {event}')
    return result


def summary_prediction(parsed):
    if not isinstance(parsed, dict) or any(k not in parsed for k in SUMMARY_FIELDS):
        raise ValueError('Summary must include all five entity lists')
    for key in SUMMARY_FIELDS:
        if not isinstance(parsed[key], list):
            raise ValueError(f'{key} must be a list')
        for item in parsed[key]:
            if not (isinstance(item, str) and item.strip() or isinstance(item, dict)
                    and isinstance(item.get('name'), str) and item['name'].strip()):
                raise ValueError(f'Invalid entity in {key}')
    for key in ('allergy', 'outcome'):
        if key in parsed and not isinstance(parsed[key], str):
            raise ValueError(f'{key} must be text')
    warnings = parsed.get('warnings', [])
    if not isinstance(warnings, list) or any(not isinstance(w, dict)
            or not isinstance(w.get('target'), str) or not isinstance(w.get('reason'), str)
            for w in warnings):
        raise ValueError('warnings must contain target/reason objects')
    return parsed


def infer(messages, validate, model, provider=None, url=None, key=None):
    limit = completion_budget(messages, config.LLM_MAX_TOKEN_BUDGET)
    caps = sorted(set(min(n, limit) for n in (8192, 12288, 16384, limit)))
    last = None
    for cap in caps:
        answer = generate(model, messages, max_tokens=cap, endpoint=provider,
                          url_override=url, key_override=key)
        try:
            return validate(extract_json(answer['text']))
        except (ValueError, LLMError) as error:
            last = error
    raise LLMError(f'Invalid response after {len(caps)} attempts: {last}')


def model_options(args):
    slot = config.BASE_SLOT or (None, None, None, None)
    model = args.model or slot[0]
    if not model:
        raise ValueError('Set BASE_MODEL or pass --model')
    return model, args.endpoint or slot[2], slot[1], slot[3]


def run_fingerprint(task, options, extra=None):
    model, provider, url, key = options
    route = resolve(model, provider, url, key)
    safe_route = {k: route[k] for k in ('provider', 'model', 'url', 'thinking')}
    settings = {name: getattr(config, name) for name in (
        'LLM_TEMPERATURE', 'LLM_THINKING', 'LLM_REPETITION_PENALTY',
        'LLM_MAX_TOKEN_BUDGET', 'LLM_RETRIES', 'LLM_TIMEOUT', 'RENDER_MAX_ABNORMAL_LABS')}
    settings['context_length'] = os.environ.get('LLM_CONTEXT_LEN', '32768')
    return digest({'task':task, 'version':RUNNER_VERSION, 'code':source_digest(),
                   'route':safe_route, 'settings':settings, 'extra':extra})


def execute(items, worker, validate, task, fingerprint, checkpoint=''):
    """Each item: id, input, context. Persist only validated successes.

    Failed items are recorded in the returned result and retried on resume.
    Reject foreign rows, inconsistent duplicates, old versions and torn files.
    """
    expected = {item['id']: {'task':task, 'run_fingerprint':fingerprint,
                            'sample_sha256':digest(item)} for item in items}
    if len(expected) != len(items):
        raise ValueError('Duplicate item IDs')
    prior = {}
    if checkpoint and Path(checkpoint).exists():
        with open(checkpoint, encoding='utf-8') as stream:
            for number, line in enumerate(stream, 1):
                if not line.endswith('\n'):
                    raise ValueError(f'Checkpoint line {number} is unterminated')
                try:
                    row = json.loads(line)
                except ValueError as error:
                    raise ValueError(f'Malformed checkpoint line {number}') from error
                item_id = row.get('id') if isinstance(row, dict) else None
                if item_id not in expected or any(row.get(k) != v for k,v in expected[item_id].items()):
                    raise ValueError(f'Incompatible checkpoint line {number}; use a new path')
                pred = validate(row.get('prediction'))
                if item_id in prior and prior[item_id] != pred:
                    raise ValueError(f'Conflicting checkpoint line {number}')
                prior[item_id] = pred
    pending = [item for item in items if item['id'] not in prior]
    if checkpoint:
        Path(checkpoint).parent.mkdir(parents=True, exist_ok=True)

    def work(item):
        return validate(worker(item))

    def progress(i, prediction, error, done, total):
        if checkpoint and error is None:
            item_id = pending[i]['id']
            with open(checkpoint, 'a', encoding='utf-8') as stream:
                stream.write(json.dumps({'id':item_id, 'prediction':prediction,
                                         **expected[item_id]}, ensure_ascii=False)+'\n')
        print(f'[{done}/{total}] {"ok" if error is None else "failed"}', flush=True)

    completed = dict(zip((item['id'] for item in pending), run_concurrent(work, pending, progress)))
    rows = []
    for item in items:
        item_id = item['id']
        pred, err = (prior[item_id], None) if item_id in prior else completed[item_id]
        rows.append({'id':item_id, 'status':'ok' if err is None else 'failed',
                     'prediction':pred if err is None else None,
                     'error':None if err is None else type(err).__name__ + ': ' + str(err)[:200],
                     **expected[item_id]})
    return rows


def save_result(path, metrics, rows):
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as stream:
            json.dump({'metrics':metrics, 'predictions':rows}, stream, ensure_ascii=False, indent=2)
