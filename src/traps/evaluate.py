"""Paired clean/perturbed single-model evaluation with descriptive target states."""
import argparse
from collections import Counter, defaultdict
import json
from src.runners import prompts
from src.runners.common import (infer, summary_prediction, model_options,run_fingerprint,execute,save_result)
from src.traps.scoring import assess, SCORE_VERSION, STATES
from src import config


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
    options=model_options(args); fingerprint=run_fingerprint('traps',options,SCORE_VERSION)
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
    paired=[]
    for trap in traps:
        clean=by_id[trap['trap_id']+':clean']; tampered=by_id[trap['trap_id']+':tampered']
        stats=counts[trap['type']]; stats['n']+=1
        detail={'trap_id':trap['trap_id'],'type':trap['type']}
        for condition,row in [('clean',clean),('tampered',tampered)]:
            if row['status']!='ok':
                stats[condition+'_failed']+=1; detail[condition]={'state':'failed'}
                continue
            score=assess(trap,row['prediction'])
            if condition=='clean':
                warning=any(t['state']=='explicit_recognition' for t in score['targets'])
                stats['clean_valid']+=1;stats['clean_targeted_warning']+=warning
                detail['clean']={'targeted_warning':warning}
            else:
                stats['tampered_valid']+=1;stats[score['state']]+=1;detail['tampered']=score
        if clean['status']==tampered['status']=='ok': stats['paired_valid']+=1
        paired.append(detail)
    per_type={}
    for kind,stats in counts.items():
        valid=stats['tampered_valid']; clean_valid=stats['clean_valid']
        per_type[kind]={**dict(stats),**{s:stats[s] for s in STATES},
            'explicit_recognition_rate':stats['explicit_recognition']/valid if valid else None,
            'incorrect_adoption_rate':stats['incorrect_adoption']/valid if valid else None,
            'clean_targeted_warning_rate':stats['clean_targeted_warning']/clean_valid if clean_valid else None}
    metrics={'model':options[0],'score_version':SCORE_VERSION,'n':len(traps),'response_count':len(rows),
             'fails':sum(r['status']!='ok' for r in rows),'per_type':per_type,
             'run_fingerprint':fingerprint,'denominator':'per-condition valid responses; failures and unresolved states reported separately'}
    # Per-case predictions remain private, enabling re-scoring without rerunning.
    for row in rows:
        row['assessment']=next(d for d in paired if row['id'].rsplit(':',1)[0]==d['trap_id'])
    save_result(args.out,metrics,rows)

if __name__=='__main__': main()
