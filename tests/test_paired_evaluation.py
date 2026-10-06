import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from generate_data import generate
from rsi.common import DEFAULTS, read_jsonl, write_json, write_jsonl, file_hash
from rsi import paired_evaluation
from rsi.tasks import reference_answer
from rsi.paired_evaluation import (audit_matching, freeze, run, analyze, tree_hash,
                                   prompt_errors, paired_interval)


class PairedEvaluationTests(unittest.TestCase):
    def fixture(self):
        tasks = {str(i): {'task':'arithmetic', 'expression':'1+1', 'difficulty':'0'} for i in range(16)}
        pool = [{'id':str(i), 'task_id':str(i), 'response':'2' if i<12 else '3'} for i in range(16)]
        return tasks,pool,{'R':[str(i) for i in range(16)],'S':[str(i) for i in range(16)]}

    def test_matching_recomputes_truth_and_denominator(self):
        tasks,pool,sub = self.fixture()
        pool.append({'id':'extra','task_id':'0','response':'4'})
        report = audit_matching(pool,sub,tasks,lambda s:2)
        self.assertEqual(report['arms']['R']['final_FPR'],4/5)
        self.assertEqual(report['arms']['R']['precision'],.75)
        # Identical R/S is legal; do not force a difference.
        self.assertTrue(report['passed'])

    def test_matching_rejects_bad_roles_and_length(self):
        tasks,pool,sub = self.fixture()
        sub['R'].append('0')
        with self.assertRaisesRegex(ValueError,'duplicate'):
            audit_matching(pool,sub,tasks,lambda s:2)
        sub['R'].pop()
        pool.append({'id':'long','task_id':'15','response':'33'})
        sub['S'][-1]='long'
        with self.assertRaisesRegex(ValueError,'token'):
            audit_matching(pool,sub,tasks,len)

    def test_matching_follows_the_run_s_ratio_and_tolerance(self):
        tasks = {str(i): {'task': 'arithmetic', 'expression': '1+1', 'difficulty': '0'} for i in range(16)}
        pool = [{'id': str(i), 'task_id': str(i), 'response': '2' if i < 14 else '3'} for i in range(16)]
        sub = {'R': [str(i) for i in range(16)], 'S': [str(i) for i in range(16)]}
        self.assertTrue(audit_matching(pool, sub, tasks, lambda s: 2, error_fraction=0.125)['passed'])  # 14 + 2
        with self.assertRaisesRegex(ValueError, 'C:E'):
            audit_matching(pool, sub, tasks, lambda s: 2)
        pool.append({'id': 'long', 'task_id': '15', 'response': '33'})
        sub['S'][-1] = 'long'   # S's error one token longer than R's
        self.assertTrue(audit_matching(pool, sub, tasks, len, error_fraction=0.125, token_tolerance=1)['passed'])
        with self.assertRaisesRegex(ValueError, 'token'):
            audit_matching(pool, sub, tasks, len, error_fraction=0.125)

    def test_prompt_average_and_paired_interval(self):
        rows=[{'task_id':'a','sample':i,'correct':i==0} for i in range(4)]
        self.assertEqual(prompt_errors(rows),{'a':.75})
        self.assertEqual(paired_interval({'a':.75},{'a':.25},50,0),(.5,.5,.5))
        with self.assertRaises(ValueError):
            prompt_errors(rows+rows)

    def test_end_to_end_mock_and_tamper(self):
        for kind in ("arithmetic", "graph"):
            with self.subTest(task=kind):
                self.exercise_pipeline(kind)

    def test_the_audit_receives_the_run_s_matching_settings(self):
        """run() hands the frozen config's matching section to audit_matching; the recorder then audits
        with the defaults, since this fixture's subsets are 3:1 at tolerance 0."""
        real, seen = paired_evaluation.audit_matching, []

        def recorder(*args, **kwargs):
            seen.append((kwargs.get('error_fraction'), kwargs.get('token_tolerance')))
            return real(*args)
        with mock.patch.object(paired_evaluation, 'audit_matching', side_effect=recorder):
            self.exercise_pipeline('arithmetic', matching={'error_fraction': 0.125, 'token_tolerance': 2})
        self.assertTrue(seen)
        self.assertEqual(set(seen), {(0.125, 2)})

    def exercise_pipeline(self, kind, matching=None):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            generate(root/'data',kind,2027,1,32,4,4,8,4)
            cfg=copy.deepcopy(DEFAULTS); cfg['backend']='mock'
            if matching:
                cfg['matching']=matching
            write_json(root/'config.json',cfg)
            freeze(root/'config.json',root/'data',root/'protocol.json',4,100,1,50,42)
            tasks=read_jsonl(root/'data/train_001.jsonl')[:16]
            pool=[]
            for i,task in enumerate(tasks):
                truth=reference_answer(task)
                pool.append({'id':str(i),'task_id':task['id'],'response':truth if i<12 else (str(int(truth)+1) if kind=='arithmetic' else '[]')})
            write_jsonl(root/'pool.jsonl',pool)
            ids=[r['id'] for r in pool]
            write_json(root/'matched.json',{'K':16,'per_block':{'b0':{'R':ids,'S':ids}}})
            for arm in ('initial','R','S'):
                write_json(root/arm/'mock.json',{'p':.55})
            write_json(root/'initial/shared_adapter.json',{'base_model':cfg['model'],'base_revision':cfg['revision'],'parameter_hash':'shared'})
            write_json(root/'one_step.json',{'block':'b0','seed':0,'arms':{'R':{},'S':{}}})
            block={'id':'b0','seed':0,'base_model':cfg['model'],'base_revision':cfg['revision'],
                   'adapters':{a:a for a in ('initial','R','S')},
                   'adapter_hashes':{a:tree_hash(root/a) for a in ('initial','R','S')},
                   'training':{a:{'completed':True,'optimizer_steps':1,'initial_parameter_hash':'shared'} for a in ('R','S')}}
            for kind,name in [('pool','pool.jsonl'),('matched','matched.json'),('one_step','one_step.json')]:
                block[kind]=name; block[kind+'_hash']=file_hash(root/name)
            write_json(root/'handoff.json',{'blocks':[block]})
            run(root/'protocol.json',root/'handoff.json',root/'eval')
            with self.assertRaisesRegex(ValueError,'Demo'):
                analyze(root/'eval',root/'analysis')
            results=analyze(root/'eval',root/'analysis',True)
            self.assertTrue(all(r['mean']==0 for r in results if r['contrast']=='S-R'))
            self.assertTrue((root/'analysis/metrics_per_block.csv').exists())
            with self.assertRaisesRegex(ValueError,'fresh'):
                run(root/'protocol.json',root/'handoff.json',root/'eval')
            write_json(root/'R/mock.json',{'p':.9})
            with self.assertRaisesRegex(ValueError,'hash'):
                run(root/'protocol.json',root/'handoff.json',root/'eval2')


if __name__=='__main__':
    unittest.main()
