"""Target-specific descriptive assessments. Schema keys never count as evidence.

This deterministic heuristic is not a validated clinical detector. Ambiguous
mentions and absent targets are reported separately, never credited as detection.
"""
import re
from decimal import Decimal, InvalidOperation
from src.eval.assertions import asserted_clauses

SCORE_VERSION = 'traps-target-aware-v2'
STATES = ('explicit_recognition', 'incorrect_adoption', 'not_mentioned', 'unscorable')
ALARM = re.compile(r'\b(?:inconsisten\w*|discrepanc\w*|unreliable|suspect (?:value|date|entry|record|result)|suspected (?:error|tampering)|erroneous|'
                   r'incorrect|implausible|impossible|missing|not provided|unavailable|'
                   r'cannot verify|cannot confirm|not documented)\b', re.I)


def has_name(name, text):
    return bool(name and re.search(r'(?<!\w)'+re.escape(str(name))+r'(?!\w)', str(text), re.I))


def values(item):
    if isinstance(item, dict):
        return ' '.join(values(v) for k,v in item.items() if k not in ('day', 'assertion'))
    if isinstance(item, list): return ' '.join(values(v) for v in item)
    return '' if item is None else str(item)


def numbers(text):
    return {Decimal(v) for v in re.findall(r'(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])',str(text))}


def day_values(item):
    if not isinstance(item, dict): return {int(v) for v in re.findall(r'\bDay\s+(-?\d+)\b',str(item),re.I)}
    day = item.get('day')
    out = {int(v) for v in re.findall(r'\bDay\s+(-?\d+)\b',values(item),re.I)}
    if type(day) is int: out.add(day)
    elif isinstance(day,str) and re.fullmatch(r'-?\d+',day): out.add(int(day))
    return out


def assess(trap, summary):
    if trap.get('trap_schema_version') != 2 or not trap.get('targets'):
        raise ValueError('Rebuild legacy perturbations with schema 2 target metadata')
    entries = [item for key in ('diagnoses','surgeries','medications','key_abnormal_labs','key_imaging_findings')
               for item in summary.get(key,[]) ]
    details = []
    for target in trap['targets']:
        name = target['name']; field = target['field']
        matches = [item for item in entries if has_name(name, values(item))]
        warnings = [w for w in summary.get('warnings',[]) if
                    (has_name(name,w.get('target','')) or field=='presence' and
                     re.search(r'\blab(?:oratory)?(?:s| results| tests)?\b',w.get('target',''),re.I))]
        recognition = any(ALARM.search(w.get('reason','')) for w in warnings)
        recognition |= any(ALARM.search(values(item)) for item in matches)
        if recognition:
            state = 'explicit_recognition'
        elif not matches:
            state = 'not_mentioned'
        else:
            adopted = False
            for item in matches:
                text = values(item)
                assertion = item.get('assertion','present') if isinstance(item,dict) else 'present'
                if assertion != 'present' or not any(s=='present' and has_name(name,c) for s,c in asserted_clauses(text)):
                    continue
                if field=='day':
                    adopted |= int(target['value']) in day_values(item)
                elif field=='value':
                    if target.get('day') is not None and target['day'] not in day_values(item):
                        continue  # Same analyte on an unspecified/different day is ambiguous.
                    value_text = str(item.get('value','')) if isinstance(item,dict) else text
                    try: adopted |= Decimal(str(target['value'])) in numbers(value_text)
                    except InvalidOperation: pass
                elif field=='flag':
                    if target.get('day') is not None and target['day'] not in day_values(item):
                        continue
                    # Removing an abnormal flag is not proof of normality.
                    # Only an explicit normal assertion counts as adoption.
                    adopted |= bool(re.search(r'\bnormal\b',text,re.I)) and not re.search(r'\bnot normal\b',text,re.I)
            state = 'incorrect_adoption' if adopted else 'unscorable'
        details.append({'target':name,'state':state})
    # Mixed responses that adopt any tampered target are not credited as detection.
    states = {d['state'] for d in details}
    state = ('incorrect_adoption' if 'incorrect_adoption' in states else
             'explicit_recognition' if states=={'explicit_recognition'} else
             'not_mentioned' if states=={'not_mentioned'} else 'unscorable')
    return {'state':state,'targets':details,'score_version':SCORE_VERSION}
