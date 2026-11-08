"""Collect paired clean/perturbed responses and execution counts without scoring."""
import argparse
from collections import Counter, defaultdict
import json
from src.runners import prompts
from src.runners.common import (infer, summary_prediction, model_options,run_fingerprint,execute,save_result)
from src import config

OUTPUT_VERSION = 'traps-responses-v1'


def select_traps(traps,n):
    if n < 0: raise ValueError('--n must be nonnegative')
    if not n or n >= len(traps): return traps
    groups = defaultdict(list)
    for trap in traps: groups[trap['type']].append(trap)
    ordered = [rows[i] for i in range(max(map(len,groups.values()))) for rows in groups.values() if i < len(rows)]
    return ordered[:n]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model'); ap.add_argument('--endpoint')
    ap.add_argument('--n',type=int,default=0); ap.add_argument('--out',default='')
    ap.add_argument('--ckpt',default=''); ap.add_argument('--traps',default='')
    args=ap.parse_args()
    with open(args.traps or config.TRAPS_FILE,encoding='utf-8') as stream:
        traps=select_traps([json.loads(line) for line in stream if line.strip()],args.n)
    options=model_options(args); fingerprint=run_fingerprint('traps',options,OUTPUT_VERSION)
    items=[]
    for trap in traps:
        if trap.get('trap_schema_version')!=2 or not trap.get('targets') or not trap.get('clean_input') or not trap.get('tampered_input'):
            raise ValueError('Rebuild legacy traps with paired inputs and target metadata (schema 2)')
        context={k:v for k,v in trap.items() if k not in ('clean_input','tampered_input')}
        for condition,key in [('clean','clean_input'),('tampered','tampered_input')]:
            items.append({'id':trap['trap_id']+':'+condition,'input':trap[key],
                          'context':{'condition':condition,**context}})
    def work(item):
        messages=[{'role':'system','content':prompts.B1_SYSTEM},
                  {'role':'user','content':prompts.B1_USER.format(timeline=item['input'])}]
        return infer(messages,summary_prediction,*options)
    rows=execute(items,work,summary_prediction,'traps',fingerprint,args.ckpt)
    by_id={row['id']:row for row in rows}; counts=defaultdict(Counter)
    for trap in traps:
        clean=by_id[trap['trap_id']+':clean']; tampered=by_id[trap['trap_id']+':tampered']
        stats=counts[trap['type']]; stats['n']+=1
        for condition,row in [('clean',clean),('tampered',tampered)]:
            stats[condition+('_valid' if row['status']=='ok' else '_failed')]+=1
            row.update(trap_id=trap['trap_id'], type=trap['type'],
                       condition=condition, targets=trap['targets'])
        if clean['status']==tampered['status']=='ok': stats['paired_valid']+=1
    count_fields=('n','clean_valid','clean_failed','tampered_valid','tampered_failed','paired_valid')
    per_type={kind:{field:stats[field] for field in count_fields} for kind,stats in counts.items()}
    metrics={'model':options[0],'output_version':OUTPUT_VERSION,'n':len(traps),'response_count':len(rows),
             'fails':sum(r['status']!='ok' for r in rows),'per_type':per_type,
             'run_fingerprint':fingerprint}
    # Keep model responses and target metadata; no semantic classification or scores.
    save_result(args.out,metrics,rows)

if __name__=='__main__': main()
