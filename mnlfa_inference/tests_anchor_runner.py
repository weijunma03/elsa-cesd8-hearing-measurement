"""Synthetic-only runner contract tests; no real loader, data or fits."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import copy
import importlib.util
import json
import tempfile
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('anchors_under_test', ROOT/'run_anchors.py')
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)
checks = []


def check(name, passed):
    checks.append({'name': name, 'passed': bool(passed)})
    if not passed:
        raise AssertionError(name)


def rejected(name, call):
    try:
        call()
    except (ValueError, RuntimeError):
        check(name, True)
    else:
        check(name, False)


def main():
    check('import_does_not_start_analysis', not r.RUN_STARTED and r.START is None and not r.ATTEMPTS)
    check('real_loader_not_imported', 'anchor_covariate_loader' not in sys.modules)
    plan = r.read_json(ROOT/'analysis_plan.json')
    cfg = r.validate_anchor_plan(plan)
    for key, value in [('stage','hearing_only'), ('parameters_each',54), ('quadrature_pairs',[[81,121]])]:
        bad = copy.deepcopy(plan)
        bad['anchors'][key] = value
        rejected('reject_changed_'+key, lambda bad=bad:r.validate_anchor_plan(bad))
    bad = copy.deepcopy(plan)
    bad['anchors']['runs'].reverse()
    rejected('reject_reordered_models', lambda:r.validate_anchor_plan(bad))
    rng = np.random.default_rng(20261025)
    x = np.column_stack([np.arange(24)%2, rng.integers(-2,3,24)/2, np.arange(24)//2%2])
    y = rng.integers(0,2,(24,8)).astype(float)
    w = rng.uniform(.3,2.1,24)
    grouped = r.group_rows(y,x,w)
    profiles, inverse = np.unique(x[:,1:],axis=0,return_inverse=True)
    pw = np.bincount(inverse,weights=w)
    prior = r.Model('slope_adjusted',7)
    prior_t = prior.initial(grouped)
    for run in cfg['runs']:
        model = r.Model('slope_adjusted',7,anchors=run['anchors'])
        t, metadata = model.transfer_start(prior_t,prior)
        check(run['id']+'_60_parameters', model.p==60)
        check(run['id']+'_six_zero_new_names', len(metadata['new_zero_names'])==6 and not metadata['dropped_names'] and
              all(t[model.names.index(name)]==0 for name in metadata['new_zero_names']))
        check(run['id']+'_five_free_targets', len(model.nonanchors)==5 and set(range(4,8)).issubset(model.nonanchors))
        summary = {'model':run['id'],'released_item':'i'+str(run['released']+1),'numerically_admissible':True}
        rows = r.scientific_contrasts(model,t,summary,profiles,pw)
        check(run['id']+'_five_scientific_rows', len(rows)==5 and sum(row['original_target'] for row in rows)==4 and
              all(row['contrast_pp'] is not None for row in rows))
        summary['numerically_admissible'] = False
        rows = r.scientific_contrasts(model,t,summary,profiles,pw)
        check(run['id']+'_failed_contrasts_suppressed', all(row['contrast_pp'] is None and row['contrast'] is None for row in rows))
        residuals = r.local_diagnostics(model,t,summary,x,y,w)
        check(run['id']+'_72_aggregate_residuals', len(residuals)==72 and sum(row['kind']=='pair' for row in residuals)==56)
        check(run['id']+'_residual_arithmetic', all(abs(row['residual']-(row['observed']-row['predicted']))<1e-14 for row in residuals))
        check(run['id']+'_deterministic_perturbation', np.array_equal(r.perturb(t,model),r.perturb(t,model)))
        check(run['id']+'_perturbation_definition', np.array_equal(r.perturb(t,model),np.clip(t+.08*np.sin(np.arange(1,61)),np.array(model.bounds)[:,0]+.001,np.array(model.bounds)[:,1]-.001)))
    with tempfile.TemporaryDirectory(prefix='anchor_runner_check_',dir=ROOT) as temporary:
        base = Path(temporary)
        stage, protected = base/'stage', base/'protected.txt'
        stage.mkdir()
        (stage/'code.py').write_text('synthetic fixture',encoding='utf-8')
        protected.write_text('protected synthetic fixture',encoding='utf-8')
        frozen = {'files':{'code.py':r.sha(stage/'code.py')},'protected':{str(protected.resolve()):r.sha(protected)}}
        r.dump(stage/'freeze.json',frozen)
        check('valid_freeze_matches', r.validate_freeze(stage)==frozen)
        (stage/'code.py').write_text('changed',encoding='utf-8')
        rejected('reject_local_hash_mutation', lambda:r.validate_freeze(stage))
        (stage/'code.py').write_text('synthetic fixture',encoding='utf-8')
        protected.write_text('changed protected',encoding='utf-8')
        rejected('reject_protected_hash_mutation', lambda:r.validate_freeze(stage))
        frozen['files']={'../protected.txt':r.sha(protected)}
        r.dump(stage/'freeze.json',frozen)
        rejected('reject_stage_path_traversal', lambda:r.validate_freeze(stage))
        duplicate = base/'duplicate'
        duplicate.mkdir()
        (duplicate/'progress.json').write_text('must be unchanged',encoding='utf-8')
        with patch.object(r,'OUT',duplicate):
            rejected('reject_duplicate_run',r.main)
        check('duplicate_progress_preserved',(duplicate/'progress.json').read_text()=='must be unchanged')
        rejected('network_socket_guard',lambda:r.guard('socket.connect',()))
        check('nonnetwork_event_allowed',r.guard('open',()) is None)
        # Mock optimizer plus quadratic synthetic objective exercises runner gates.
        # This performs no real or synthetic parameter fitting and reads no data.
        real_model = r.Model
        for scenario in ('passes_first','refines_then_passes','fails_both','optimizer_fails'):
            offsets = {'passes_first':{},'refines_then_passes':{61:.001},'fails_both':{61:.001,81:.0001},'optimizer_fails':{}}[scenario]
            class MockModel(real_model):
                def fg(self,t,_):
                    return float(.5*np.dot(t,t)+offsets.get(self.q,0.)),np.asarray(t).copy()
            calls = []
            def optimizer(fun,start,**kwargs):
                calls.append(np.asarray(start).copy())
                return SimpleNamespace(x=np.zeros(60),success=scenario!='optimizer_fails',message='synthetic mock',nit=1,nfev=1)
            output = base/scenario
            output.mkdir()
            with patch.object(r,'Model',MockModel),patch.object(r,'minimize',optimizer),patch.object(r,'OUT',output),patch.object(r,'ATTEMPTS',[]),patch.object(r,'progress',lambda *a,**kw:None):
                run = cfg['runs'][0]
                m,cm,t,summary = r.fit_one(run,cfg,None,np.zeros(60),{'synthetic_mock':True},np.array([[0.,0.]]),np.ones(1))
                attempts = r.read_json(output/'attempts.json')
                expected_calls = 4 if scenario in ('refines_then_passes','fails_both') else 2
                check(scenario+'_fixed_attempt_count',len(calls)==len(attempts)==expected_calls)
                check(scenario+'_six_checks',len(summary['checks'])==6)
                check(scenario+'_gate_outcome',summary['numerically_admissible']==(scenario in ('passes_first','refines_then_passes')))
                check(scenario+'_hessian_saved',np.allclose(summary['hessian_symmetric'],np.eye(60)) and len(summary['parameter_names'])==60)
                check(scenario+'_starts_and_gradients_saved',all(len(a['initial_parameters'])==len(a['gradient'])==60 for a in attempts))
                check(scenario+'_deterministic_starts',all(np.array_equal(calls[j],r.perturb(np.zeros(60),m)) for j in range(1,len(calls),2)))
                check(scenario+'_q_limits',summary['fit_q'] in (61,81) and summary['check_q'] in (81,121))
    result = {'status':'PASS','passed':True,'checks_passed':len(checks),'checks':checks,
              'real_data_read':False,'real_or_synthetic_optimizer_run':False,
              'methods':'Synthetic helper calculations; mocked optimizer/objective for gate orchestration; temporary mock freeze and duplicate-run files.',
              'run_anchors_sha256':r.sha(ROOT/'run_anchors.py')}
    r.dump(ROOT/'anchor_runner_synthetic_checks.json',result)
    print(json.dumps({'status':result['status'],'checks_passed':len(checks),'run_anchors_sha256':result['run_anchors_sha256']}))


if __name__=='__main__':
    main()
