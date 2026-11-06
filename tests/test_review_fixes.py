"""Regression tests for the public-source review; only synthetic/mocked inputs."""
import contextlib
from datetime import timedelta
import io
import json
import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import Mock, patch
import pandas as pd

from src import config
from src.eval.metrics import b1_entity_metrics, b3_multilabel_metrics
from src.llm import client
from src.llm.budget import completion_budget
from src.mimic import extract
from src.mimic.timeline import build_one
from src.runners import common, run_b1, run_b3
from src.tasks import b3
from src.traps import build, evaluate
from src.traps.scoring import assess
from test_b3_temporal import synthetic_tables, ADMIT


def summary(**fields):
    return {**{key:[] for key in common.SUMMARY_FIELDS}, **fields}


class Predictions(unittest.TestCase):
    def test_string_zeros_map_to_zero(self):
        self.assertEqual(common.binary_prediction({e:'0' for e in b3.EVENTS},b3.EVENTS),
                         {e:0 for e in b3.EVENTS})

    def test_invalid_binary_schema(self):
        for invalid in ({}, {'discharge':0}, None, [],
                        *({e:value for e in b3.EVENTS} for value in (None,True,0.0,'no','0 or 1',-1))):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                common.binary_prediction(invalid,b3.EVENTS)

    def test_legitimate_all_positive_and_all_negative_are_valid(self):
        for value in (0,1):
            prediction={e:value for e in b3.EVENTS}
            self.assertEqual(common.binary_prediction({'predictions':prediction},b3.EVENTS),prediction)

    def test_bad_model_answer_does_not_return_last_coercion(self):
        with patch.object(common,'generate',return_value={'text':'{}'}):
            with self.assertRaises(client.LLMError):
                run_b3.run_baseline('synthetic','synthetic',1)

    def test_empty_summary_object_is_invalid_but_explicit_empty_lists_valid(self):
        with self.assertRaises(ValueError): common.summary_prediction({})
        self.assertEqual(common.summary_prediction(summary()),summary())

    def test_failed_b3_not_successful_all_negative(self):
        doc=build_one(1,synthetic_tables()); sample={'admission_id':'A-1','labels':{e:0 for e in b3.EVENTS}}
        with tempfile.TemporaryDirectory() as td:
            output=Path(td,'result.json')
            with patch.object(run_b3,'load_samples',return_value=([sample],{'A-1':doc})), \
                 patch.object(run_b3,'model_options',return_value=('synthetic',None,None,None)), \
                 patch.object(run_b3,'run_fingerprint',return_value='test'), \
                 patch.object(common,'generate',return_value={'text':'{}'}), \
                 patch('sys.argv',['b3','--X','1','--out',str(output)]), contextlib.redirect_stdout(io.StringIO()):
                run_b3.main()
            result=json.loads(output.read_text())
            self.assertEqual(result['metrics']['fails'],1)
            self.assertEqual(result['metrics']['n_valid'],0)
            self.assertIsNone(result['metrics']['exact_match'])
            self.assertIsNone(result['predictions'][0]['prediction'])

    def test_empty_and_misaligned_metrics(self):
        self.assertIsNone(b3_multilabel_metrics([],[],b3.EVENTS)['macro_f1'])
        with self.assertRaises(ValueError): b3_multilabel_metrics([{}],[],b3.EVENTS)

    def test_b3_entry_filters_input_keeps_labels_and_resumes(self):
        doc=build_one(1,synthetic_tables(ADMIT+timedelta(days=4),stop_day=2))
        sample={'admission_id':'A-1','labels':{e:int(e=='antibiotic_stop') for e in b3.EVENTS}}
        with tempfile.TemporaryDirectory() as td,contextlib.redirect_stdout(io.StringIO()):
            out=Path(td,'result.json');ckpt=Path(td,'check.jsonl')
            with patch.object(run_b3,'load_samples',return_value=([sample],{'A-1':doc})), \
                 patch.object(run_b3,'model_options',return_value=('synthetic',None,None,None)), \
                 patch.object(run_b3,'run_fingerprint',return_value='test'), \
                 patch.object(run_b3,'run_baseline',return_value={e:0 for e in b3.EVENTS}) as model, \
                 patch('sys.argv',['b3','--X','1','--out',str(out),'--ckpt',str(ckpt)]):
                run_b3.main()
                text=model.call_args.args[1]
                for value in ('stop Day 2','TEST_LAB','TEST_ORGANISM','TEST_FINDING'): self.assertNotIn(value,text)
                result=json.loads(out.read_text())
                self.assertEqual(result['predictions'][0]['y_true']['antibiotic_stop'],1)
                model.reset_mock();run_b3.main();model.assert_not_called()

    def test_b1_entry_invalid_response_is_counted_and_not_scored(self):
        case={'admission_id':'A-1','input':'synthetic','entities':[{'type':'discharge_diagnosis','name':'Sepsis'}]}
        with tempfile.TemporaryDirectory() as td,contextlib.redirect_stdout(io.StringIO()):
            out=Path(td,'result.json')
            with patch.object(run_b1,'load_cases',return_value=[case]), \
                 patch.object(run_b1,'model_options',return_value=('synthetic',None,None,None)), \
                 patch.object(run_b1,'run_fingerprint',return_value='test'), \
                 patch.object(common,'generate',return_value={'text':'{}'}), \
                 patch('sys.argv',['b1','--out',str(out)]): run_b1.main()
            result=json.loads(out.read_text())
            self.assertEqual(result['metrics']['fails'],1)
            self.assertEqual(result['metrics']['n_valid'],0)
            self.assertIsNone(result['metrics']['entity_recall_mean'])


class Checkpoints(unittest.TestCase):
    def test_all_tasks_bound_to_input_model_and_selected_set(self):
        for task in ('B1','B3','traps'):
            with self.subTest(task=task), tempfile.TemporaryDirectory() as td, contextlib.redirect_stdout(io.StringIO()):
                path=str(Path(td,'checkpoint.jsonl'))
                items=[{'id':'test','input':'old','context':{'reference':'old'}}]
                validator=lambda p: common.binary_prediction(p,['event'])
                worker=Mock(return_value={'event':0})
                common.execute(items,worker,validator,task,'model-one',path)
                worker.reset_mock()
                common.execute(items,worker,validator,task,'model-one',path)
                worker.assert_not_called()
                for candidate,fingerprint in (([{**items[0],'input':'new'}],'model-one'),
                        ([{**items[0],'context':{'reference':'new'}}],'model-one'),
                        (items,'model-two'),([], 'model-one')):
                    with self.assertRaisesRegex(ValueError,'Incompatible'):
                        common.execute(candidate,worker,validator,task,fingerprint,path)
                worker.assert_not_called()

    def test_invalid_checkpoint_prediction_and_torn_row(self):
        with tempfile.TemporaryDirectory() as td, contextlib.redirect_stdout(io.StringIO()):
            path=Path(td,'checkpoint.jsonl'); items=[{'id':'1','input':'test','context':{}}]
            validator=lambda p: common.binary_prediction(p,['event'])
            common.execute(items,lambda i:{'event':0},validator,'B3','run',str(path))
            valid=path.read_text()
            row=json.loads(valid);row['prediction']={}
            for content in (json.dumps(row)+'\n', valid.rstrip('\n'), '{broken\n'):
                path.write_text(content)
                with self.assertRaises(ValueError):
                    common.execute(items,Mock(),validator,'B3','run',str(path))

    def test_failed_case_retried_and_not_cached_as_zero(self):
        with tempfile.TemporaryDirectory() as td, contextlib.redirect_stdout(io.StringIO()):
            path=str(Path(td,'check.jsonl'));items=[{'id':'1','input':'test','context':{}}]
            validator=lambda p: common.binary_prediction(p,['event'])
            rows=common.execute(items,Mock(side_effect=client.LLMError('offline')),validator,'B3','run',path)
            self.assertEqual(rows[0]['status'],'failed')
            worker=Mock(return_value={'event':0})
            self.assertEqual(common.execute(items,worker,validator,'B3','run',path)[0]['status'],'ok')
            worker.assert_called_once()


class ExactLabels(unittest.TestCase):
    def test_discharge_before_at_and_after_boundary(self):
        for seconds,expected in ((-1,0),(0,1),(1,1)):
            tables=synthetic_tables(); end=ADMIT+timedelta(days=4,seconds=seconds)
            tables['cohort']['dischtime']=end
            tables['cohort']['los_days']=4  # intentionally rounded source field
            doc=build_one(1,tables)
            self.assertEqual(b3.derive_events(doc)['discharge'],[3 if seconds<0 else 4])
            with tempfile.TemporaryDirectory() as td:
                Path(td,'record.json').write_text(json.dumps(doc))
                with patch.object(b3,'TL_DIR',td): self.assertEqual(len(b3.build_samples(3)),expected)

    def test_death_uses_death_time(self):
        tables=synthetic_tables();tables['cohort']['hospital_expire_flag']=1
        tables['cohort']['deathtime']=ADMIT+timedelta(days=2)
        doc=build_one(1,tables)
        self.assertEqual(b3.derive_events(doc)['death'],[2])

    def test_missing_death_time_excluded_and_counted(self):
        tables=synthetic_tables();tables['cohort']['hospital_expire_flag']=1
        doc=build_one(1,tables)
        with tempfile.TemporaryDirectory() as td:
            Path(td,'record.json').write_text(json.dumps(doc));audit={}
            with patch.object(b3,'TL_DIR',td): self.assertEqual(b3.build_samples(1,audit=audit),[])
            self.assertEqual(audit['excluded_invalid_or_missing_death_time'],1)

    def test_died_before_cutoff_not_in_future_prediction_cohort(self):
        tables=synthetic_tables();tables['cohort']['hospital_expire_flag']=1
        tables['cohort']['deathtime']=ADMIT+timedelta(days=1)
        doc=build_one(1,tables)
        with tempfile.TemporaryDirectory() as td:
            Path(td,'record.json').write_text(json.dumps(doc));audit={}
            with patch.object(b3,'TL_DIR',td): self.assertEqual(b3.build_samples(1,audit=audit),[])
            self.assertEqual(audit['excluded_died_before_cutoff'],1)


class Assertions(unittest.TestCase):
    def recall(self,pred,ref='Sepsis'):
        return b1_entity_metrics([{'type':'discharge_diagnosis','name':ref}],{'diagnoses':[pred]})['entity_recall']

    def test_assertion_states_do_not_match_active_diagnosis(self):
        for text in ('No evidence of sepsis','Sepsis was ruled out','Possible sepsis',
                     'Cannot rule out sepsis','History of sepsis','Family history of sepsis'):
            with self.subTest(text=text): self.assertEqual(self.recall({'name':text}),0)
        for state in ('absent','historical','uncertain','family'):
            self.assertEqual(self.recall({'name':'Sepsis','assertion':state}),0)

    def test_contrast_and_cue_words(self):
        self.assertEqual(self.recall({'name':'No fever but sepsis'}),1)
        self.assertEqual(self.recall({'name':'No sepsis but pneumonia'}),0)
        self.assertEqual(self.recall({'name':'No pneumonia'},'No sepsis'),0)
        self.assertEqual(self.recall({'name':'No sepsis'},'No sepsis'),1)


class TransportBudget(unittest.TestCase):
    def call(self,provider,envelope,budget=1000):
        response=Mock();response.__enter__=Mock(return_value=response);response.__exit__=Mock(return_value=False)
        response.read.return_value=json.dumps(envelope).encode()
        route=dict(provider=provider,model='synthetic',url='http://synthetic.invalid/chat',key=None,
                   family=None,thinking='auto',display='synthetic')
        with patch.object(client,'resolve',return_value=route),patch.object(client,'_rate_gate'), \
             patch.object(client,'_key_disabled',return_value=False),patch.object(client,'_key_timeout',return_value=None), \
             patch.object(client,'_log_transcript'),patch.object(client.time,'sleep'), \
             patch.object(config,'LLM_RETRIES',1),patch.object(config,'LLM_MAX_TOKEN_BUDGET',budget), \
             patch.dict(os.environ,{'LLM_CONTEXT_LEN':'2048'}), \
             patch.object(client.urllib.request,'urlopen',return_value=response) as http:
            result=client.generate('synthetic',[{'role':'user','content':'synthetic'}],max_tokens=4000)
        return result,json.loads(http.call_args.args[0].data)

    def test_each_provider_envelope_and_budget(self):
        cases=[('ollama',{'message':{'role':'assistant','content':'{}'},'done':True}),
               ('openai',{'choices':[{'message':{'content':'{}'},'finish_reason':'stop'}]}),
               ('claude',{'content':[{'type':'text','text':'{}'}],'stop_reason':'end_turn'})]
        for provider,envelope in cases:
            result,payload=self.call(provider,envelope)
            self.assertEqual(result['text'],'{}')
            self.assertLessEqual(payload['options']['num_predict'] if provider=='ollama' else payload['max_tokens'],1000)

    def test_invalid_envelopes_fail(self):
        for provider in ('ollama','openai','claude'):
            with self.assertRaises(client.LLMError): self.call(provider,{'error':'synthetic'})

    def test_no_context_room_no_request(self):
        with patch.dict(os.environ,{'LLM_CONTEXT_LEN':'100'}),patch.object(client.urllib.request,'urlopen') as http:
            with self.assertRaises(ValueError): client.generate('synthetic',[{'content':'long'*100}],max_tokens=1000)
            http.assert_not_called()

    def test_retry_never_exceeds_request_ceiling(self):
        with self.assertRaises(client.LLMError):
            self.call('openai',{'choices':[{'message':{'content':''},'finish_reason':'length'}]})


class TrapScoring(unittest.TestCase):
    def trap(self,field='value',value='123.4',kind='value_magnitude'):
        return {'trap_schema_version':2,'type':kind,'targets':[{'name':'Creatinine','field':field,'value':value}]}

    def test_schema_keys_do_not_recognize_tampering(self):
        answer=summary(key_abnormal_labs=[{'name':'Creatinine','value':'123.4'}])
        self.assertEqual(assess(self.trap(),answer)['state'],'incorrect_adoption')
        answer['arbitrary_abnormal_key']='suspect inconsistent'
        self.assertEqual(assess(self.trap(),answer)['state'],'incorrect_adoption')

    def test_numeric_value_is_target_specific_and_exact(self):
        for answer in (summary(key_abnormal_labs=[{'name':'Sodium','value':'123.4'}]),
                       summary(key_abnormal_labs=[{'name':'Creatinine','value':'1234'}])):
            self.assertNotEqual(assess(self.trap(),answer)['state'],'incorrect_adoption')

    def test_same_analyte_wrong_or_missing_day_is_unresolved(self):
        trap=self.trap();trap['targets'][0]['day']=2
        for day in (None,3):
            answer=summary(key_abnormal_labs=[{'name':'Creatinine','value':'123.4','day':day}])
            self.assertEqual(assess(trap,answer)['state'],'unscorable')
        answer=summary(key_abnormal_labs=[{'name':'Creatinine','value':'123.4','day':2}])
        self.assertEqual(assess(trap,answer)['state'],'incorrect_adoption')

    def test_absence_and_unresolved_not_detection(self):
        self.assertEqual(assess(self.trap(),summary())['state'],'not_mentioned')
        self.assertEqual(assess(self.trap(),summary(key_abnormal_labs=[{'name':'Creatinine'}]))['state'],'unscorable')

    def test_target_warning_only(self):
        for target,expected in [('Creatinine','explicit_recognition'),('Sodium','incorrect_adoption')]:
            answer=summary(key_abnormal_labs=[{'name':'Creatinine','value':'123.4'}],
                           warnings=[{'target':target,'reason':'inconsistent value'}])
            self.assertEqual(assess(self.trap(),answer)['state'],expected)

    def test_clinical_uncertainty_is_not_tampering_recognition(self):
        answer=summary(key_abnormal_labs=[{'name':'Suspected Creatinine problem','value':'123.4'}])
        self.assertEqual(assess(self.trap(),answer)['state'],'unscorable')

    def test_numeric_and_text_days_equivalent(self):
        for item in ({'name':'Creatinine','day':3},{'name':'Creatinine','day':'3'},
                     {'name':'Creatinine Day 3'}):
            self.assertEqual(assess(self.trap('day',3,'temporal_shift'),summary(key_abnormal_labs=[item]))['state'],'incorrect_adoption')

    def test_flag_removal_does_not_mean_normal(self):
        trap=self.trap('flag','unmarked','polarity_flip')
        self.assertEqual(assess(trap,summary(key_abnormal_labs=[{'name':'Creatinine','value':'123.4'}]))['state'],'unscorable')
        self.assertEqual(assess(trap,summary(key_abnormal_labs=[{'name':'Creatinine','value':'normal'}]))['state'],'incorrect_adoption')

    def test_old_traps_rejected(self):
        with self.assertRaises(ValueError): assess({'type':'causal_swap'},summary())

    def test_omission_targeted_missing_warning(self):
        answer=summary(warnings=[{'target':'laboratory results','reason':'not provided'}])
        self.assertEqual(assess(self.trap('presence',None,'info_omission'),answer)['state'],'explicit_recognition')

    def test_order_swap_changes_dated_events_and_records_targets(self):
        text='=== Day 1 ===\n[Medication] DrugA 1 g\n=== Day 3 ===\n[Medication] DrugB 2 g'
        changed,info=build._swap_orders(text,random.Random(1))
        self.assertEqual(info['type'],'temporal_order_swap')
        self.assertEqual([t['value'] for t in info['targets']],[3,1])
        self.assertIn('=== Day 1 ===\n[Medication] DrugB',changed)

    def test_magnitude_targets_lab_not_age(self):
        text='Patient: 50.5y\n=== Day 1 ===\n[Lab] Creatinine = 1.2 [abnormal]'
        changed,info=build._mutate_value(text,random.Random(1))
        self.assertIn('50.5y',changed);self.assertEqual(info['targets'][0]['name'],'Creatinine')

    def test_paired_runner_reuses_only_compatible_outputs(self):
        trap=dict(self.trap(),trap_id='T1',clean_input='clean synthetic',tampered_input='tampered synthetic')
        with tempfile.TemporaryDirectory() as td,contextlib.redirect_stdout(io.StringIO()):
            source=Path(td,'traps.jsonl');out=Path(td,'result.json');ckpt=Path(td,'checkpoint.jsonl')
            source.write_text(json.dumps(trap)+'\n')
            args=['traps','--traps',str(source),'--out',str(out),'--ckpt',str(ckpt)]
            with patch.object(evaluate,'model_options',return_value=('synthetic',None,None,None)), \
                 patch.object(evaluate,'run_fingerprint',return_value='test'), \
                 patch.object(evaluate,'infer',return_value=summary()) as model,patch('sys.argv',args):
                evaluate.main();self.assertEqual(model.call_count,2)
                model.reset_mock();evaluate.main();model.assert_not_called()
                result=json.loads(out.read_text())
                stats=result['metrics']['per_type']['value_magnitude']
                self.assertEqual(stats['not_mentioned'],1)
                self.assertEqual(stats['explicit_recognition_rate'],0)
                trap['trap_id']='T2';source.write_text(json.dumps(trap)+'\n')
                with self.assertRaisesRegex(ValueError,'Incompatible'): evaluate.main()


class CohortPool(unittest.TestCase):
    def test_short_or_empty_pool_fails_clearly(self):
        for count in (0,1):
            with tempfile.TemporaryDirectory() as td:
                adm=synthetic_tables()['cohort'].copy().iloc[:count]
                adm['deathtime']=pd.NaT;adm['marital_status']='synthetic'
                icu=pd.DataFrame({'hadm_id':pd.Series(dtype='int64'),'stay_id':pd.Series(dtype='int64')})
                with patch.object(config,'MIMIC_RAW',td),patch.object(config,'COHORT_SIZE',2), \
                     patch.object(extract,'conn',return_value=Mock()), \
                     patch.object(extract,'copy_df',side_effect=[pd.DataFrame({'hadm_id':[1]}),adm,icu]):
                    with self.assertRaisesRegex(ValueError,'only .* eligible'): extract.main()


if __name__=='__main__': unittest.main()
