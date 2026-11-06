"""Target-annotated perturbations with paired clean inputs (schema 2)."""
import json
import os
import random
import re
from collections import Counter
from src import config

DS = config.B1_DATASET
OUT = config.TRAPS_FILE
TRAP_SCHEMA_VERSION = 2
LAB = re.compile(r'^\[Lab\]\s*(?P<name>[^=\n]+?)\s*=\s*(?P<value>-?\d+(?:\.\d+)?)')


def _records(text):
    day = None
    for index, line in enumerate(text.splitlines()):
        marker = re.match(r'^=== Day (-?\d+) ===', line)
        if marker:
            day = int(marker.group(1))
        lab = LAB.match(line)
        if lab:
            yield index, {'name':lab['name'].strip(), 'value':lab['value'], 'day':day, 'kind':'lab'}, lab
        elif line.startswith('[Medication]'):
            name = re.split(r'\s+\d', line[len('[Medication]'):].strip(), maxsplit=1)[0]
            if name:
                yield index, {'name':name, 'day':day, 'kind':'med'}, None


def _mutate_value(text, rng):
    candidates = [(i,t,m) for i,t,m in _records(text) if m and 0.05 <= abs(float(t['value'])) <= 5000]
    if not candidates: return None
    i,t,m = rng.choice(candidates)
    value = float(t['value']) * (10 if rng.random() < .5 else .1)
    new = format(value, '.10g')
    lines = text.splitlines(); line = lines[i]
    lines[i] = line[:m.start('value')] + new + line[m.end('value'):]
    return '\n'.join(lines), {'type':'value_magnitude','orig':t['value'],'tampered':new,
        'targets':[dict(t, field='value', original_value=t['value'], value=new)]}


def _flip_flag(text, rng):
    lines = text.splitlines()
    candidates = [(i,t) for i,t,m in _records(text) if m and re.search(r'\[(?:abnormal|critical)\]',lines[i])]
    if not candidates: return None
    i,t = rng.choice(candidates)
    lines[i] = re.sub(r' ?\[(?:abnormal|critical)\]', '', lines[i])
    return '\n'.join(lines), {'type':'polarity_flip','tampered':t['value'],
        'targets':[dict(t,field='flag',original_value='abnormal',value='unmarked')]}


def _shift_day(text, rng):
    lines = text.splitlines()
    groups = {}
    for _,t,_ in _records(text):
        if t['day'] is not None: groups.setdefault(t['day'],[]).append(t)
    if not groups: return None
    day = rng.choice(sorted(groups)); new = max(0,day+rng.choice([-3,-2,2,3,5]))
    if new == day: return None
    marker = f'=== Day {day} ==='
    lines = [f'=== Day {new} ===' if line == marker else line for line in lines]
    return '\n'.join(lines), {'type':'temporal_shift','orig':f'Day {day}','tampered':f'Day {new}',
        'targets':[dict(t,field='day',original_value=day,value=new) for t in groups[day]]}


def _graft(text, other_text, rng):
    lines = text.splitlines(); other_lines = other_text.splitlines()
    donors = [(i,t) for i,t,m in _records(other_text) if m and other_lines[i] not in lines]
    receivers = [(i,t) for i,t,m in _records(text) if m]
    if not donors or not receivers: return None
    i,target = rng.choice(receivers); j,donor = rng.choice(donors)
    line = other_lines[j]; lines.insert(i,line)
    return '\n'.join(lines), {'type':'fabrication_graft','tampered_line':line,
        'targets':[dict(donor,day=target['day'],field='value',original_value=None)]}


def _drop_key(text, rng):
    lines = text.splitlines(); labs = [(i,t) for i,t,m in _records(text) if m]
    if not labs: return None
    removed = {i for i,t in labs}
    return '\n'.join(l for i,l in enumerate(lines) if i not in removed), {
        'type':'info_omission','dropped':'laboratory results','n_dropped':len(removed),
        'targets':[dict(t,field='presence',original_value=t['value'],value=None) for _,t in labs]}


def _swap_orders(text, rng):
    # Swap dated medication lines, not a presumed causal relationship.
    lines = text.splitlines(); meds = [(i,t) for i,t,m in _records(text) if t['kind']=='med' and t['day'] is not None]
    pairs = [(a,b) for n,a in enumerate(meds) for b in meds[n+1:] if a[1]['day'] != b[1]['day'] and a[1]['name'] != b[1]['name']]
    if not pairs: return None
    (i,a),(j,b) = rng.choice(pairs); lines[i],lines[j] = lines[j],lines[i]
    return '\n'.join(lines), {'type':'temporal_order_swap','targets':[
        dict(a,field='day',original_value=a['day'],value=b['day']),
        dict(b,field='day',original_value=b['day'],value=a['day'])]}


def build(n_per_type=None, seed=None):
    n = config.TRAPS_N_PER_TYPE if n_per_type is None else n_per_type
    if n < 0: raise ValueError('n_per_type must be nonnegative')
    with open(DS,encoding='utf-8') as stream: rows = [json.loads(l) for l in stream if l.strip()]
    if not rows and n: raise ValueError('No cases available for perturbations')
    rng = random.Random(config.TRAPS_SEED if seed is None else seed); traps = []
    for fn in (_mutate_value,_flip_flag,_shift_day,_graft,_drop_key,_swap_orders):
        made = tries = 0
        while made < n and tries < max(300,n*3):
            case = rng.choice(rows); tries += 1
            if fn is _graft:
                donors = [r for r in rows if r['admission_id'] != case['admission_id']]
                result = _graft(case['input'],rng.choice(donors)['input'],rng) if donors else None
            else: result = fn(case['input'],rng)
            if result:
                tampered, info = result
                traps.append({'trap_schema_version':TRAP_SCHEMA_VERSION,'trap_id':f'T{len(traps)+1:03d}',
                    'admission_id':case['admission_id'],'clean_input':case['input'],'tampered_input':tampered,**info})
                made += 1
    return traps


def main():
    os.makedirs(os.path.dirname(OUT),exist_ok=True)
    traps = build()
    with open(OUT,'w',encoding='utf-8') as stream:
        for trap in traps: stream.write(json.dumps(trap,ensure_ascii=False)+'\n')
    print({'generated':len(traps),'requested_per_type':config.TRAPS_N_PER_TYPE,'per_type':dict(Counter(t['type'] for t in traps))})

if __name__ == '__main__': main()
