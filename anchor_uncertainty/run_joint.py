"""Fixed all-anchor joint uncertainty. Local only; no optimization, row outputs or retries."""
import os,sys
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib,datetime,time,importlib.util,traceback
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
import scipy
from scipy.stats import t as student_t
from engine import Model,group_rows
from variance import normalized_mean_influence
from joint import inputs,linearize,assemble,model_gate,target_gate,release_decisions,VARIANTS
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parents[1];OUT=ROOT/'results';START=time.time();RUN_STARTED=False

def guard(event,args):
    if event.startswith('socket.'):raise RuntimeError('Network disabled in participant analysis')
sys.addaudithook(guard)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def safe(obj):
    if isinstance(obj,np.ndarray):return safe(obj.tolist())
    if isinstance(obj,dict):return {k:safe(v) for k,v in obj.items()}
    if isinstance(obj,(list,tuple)):return [safe(v) for v in obj]
    if isinstance(obj,np.generic):return safe(obj.item())
    if isinstance(obj,float) and not np.isfinite(obj):return None
    return obj

def dump(p,obj):
    tmp=p.with_suffix('.json.tmp');tmp.write_text(json.dumps(safe(obj),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(p)
def progress(status,**kw):dump(OUT/'progress.json',dict(status=status,updated_utc=stamp(),elapsed_seconds=time.time()-START,**kw))
def verify():
    f=read(ROOT/'freeze.json')
    for name,h in f['files'].items():
        if sha(ROOT/name)!=h:raise RuntimeError('Frozen file changed: '+name)
    for name,h in f['protected'].items():
        if sha(name)!=h:raise RuntimeError('Protected source changed: '+name)
    return f

def run_model(job,y,x,w,wn):
    name=job['id'];t=np.asarray(job['theta']);anchors=job['anchors'];items=job['items'];started=time.time()
    def status(phase,**kw):dump(OUT/(name+'_progress.json'),dict(model=name,phase=phase,updated_utc=stamp(),elapsed_seconds=time.time()-started,**kw))
    grouped,inverse=group_rows(y,x,wn,return_inverse=True)
    output={};diagnostics={};previous=None
    for variant,q,step in [('q81',81,1e-4),('q121',121,1e-4),('q81half',81,5e-5)]:
        m=Model('slope_adjusted',q,anchors=anchors);assert m.names==job['parameter_names']
        status('scores',variant=variant,q=q)
        nll,grad=m.fg(t,grouped);ps=m.loglik_scores(t,grouped)
        assert np.array_equal(ps.x[inverse],x)
        scores=-ps.scores[inverse]
        if not np.isfinite(scores).all() or not np.isfinite(grad).all():raise FloatingPointError('nonfinite_score_or_gradient')
        score_mean=normalized_mean_influence(scores,x[:,0],w)['mean']
        values,J=inputs(m,t,x,w,items)
        A=np.empty((m.p,m.p))
        for j in range(m.p):
            h=step*max(1.,abs(t[j]));up=t.copy();down=t.copy();up[j]+=h;down[j]-=h
            A[:,j]=(m.fg(up,grouped)[1]-m.fg(down,grouped)[1])/(2*h)
            if (j+1)%5==0:status('hessian',variant=variant,q=q,column=j+1,columns=m.p)
        asym=float(np.max(np.abs(A-A.T)));max_abs=float(np.max(np.abs(A)));A=(A+A.T)/2
        finite=all(np.isfinite(v).all() for v in [A,grad,scores,values,J]);eig=np.linalg.eigvalsh(A) if finite else np.array([-1])
        detail=dict(model=name,variant=variant,q=q,step=step,nll=float(nll),finite=finite,
                    gradient=grad,gradient_max=float(np.max(np.abs(grad))),score_identity_error=float(np.max(np.abs(grad-score_mean))),
                    hessian=A,hessian_asymmetry=asym,hessian_max_abs=max_abs,hessian_min_eigenvalue=float(eig[0]),
                    hessian_eigenvalues=eig,bound_hits=m.bounds_hits(t),parameter_names=m.names,
                    parameter_sha256=hashlib.sha256(t.astype('<f8').tobytes()).hexdigest(),target_jacobian=J,
                    diagnostic_only=True,intervals_released=False)
        res=None
        if finite and eig[0]>0:
            res=linearize(scores,x[:,0],w,A,values,J);res['items']=['i'+str(j+1) for j in items]
            detail.update(estimates=res['estimates'],newton_correction=res['correction'])
        output[variant]=res;diagnostics[variant]=detail;dump(OUT/(name+'_'+variant+'.json'),detail)
        status('variant_completed',variant=variant)
    gates=model_gate(diagnostics);status('completed',checks=gates)
    return name,output,gates

def worker_failure(name,exc,persist=True):
    if not isinstance(exc,(FloatingPointError,np.linalg.LinAlgError)):raise exc
    if persist:
        dump(OUT/(name+'_failure.json'),dict(model=name,error_type=type(exc).__name__,failure_category='numerical_worker_failure',updated_utc=stamp(),all_model_targets_held=True))
        dump(OUT/(name+'_progress.json'),dict(model=name,phase='failed',error_type=type(exc).__name__,updated_utc=stamp()))
    return name,{v:None for v in VARIANTS},{'worker_completed':False}

def main():
    global RUN_STARTED
    if (OUT/'progress.json').exists():raise RuntimeError('Prior run exists; no silent restart')
    verify();plan=read(ROOT/'analysis_plan.json');targets=plan['estimands']['targets']
    assert len(targets)==36 and plan['intervals']['family_size']==36
    RUN_STARTED=True
    progress('loading',freeze_sha256=sha(ROOT/'freeze.json'))
    spec=importlib.util.spec_from_file_location('locked_loader',WORK/'work/elsa-mnlfa-precision-2026-10-24/data_audit/audit_covariates.py')
    loader=importlib.util.module_from_spec(spec);spec.loader.exec_module(loader)
    frame,audit=loader.load_covariate_frame('lex_first');alternate,_=loader.load_covariate_frame('lex_last')
    assert frame.drop(columns='stratum').equals(alternate.drop(columns='stratum'))
    assert audit['source_sha256']==plan['data']['source_sha256']
    d=frame.loc[frame.domain.eq(1)].copy()
    assert len(frame)==8152 and len(d)==7314 and d.hh.nunique()==5748
    assert frame.groupby('hh').stratum.nunique().max()==alternate.groupby('hh').stratum.nunique().max()==1
    hh=frame[['hh','stratum']].drop_duplicates('hh').sort_values('hh').reset_index(drop=True)
    hh2=alternate[['hh','stratum']].drop_duplicates('hh').sort_values('hh').reset_index(drop=True)
    assert np.array_equal(hh.hh,hh2.hh) and len(hh)==6217 and hh.stratum.nunique()==9
    assert (hh.stratum!=hh2.stratum).sum()==3
    hi=pd.Index(hh.hh).get_indexer(d.hh);assert (hi>=0).all()
    strata=hh.stratum.to_numpy();tie=hh2.stratum.to_numpy()
    x=d[['g','age10','female']].to_numpy(float);y=d[[f'i{j}' for j in range(1,9)]].to_numpy(float)
    w=d.w.to_numpy(float,copy=True);wn=w.copy()
    assert np.isfinite(x).all() and np.isin(y,[0,1]).all() and np.isfinite(w).all() and (w>0).all()
    for g,n in [(0,5241),(1,2073)]:
        ix=x[:,0]==g;assert ix.sum()==n;wn[ix]*=n/wn[ix].sum()
    dump(OUT/'design_audit.json',dict(domain_n=7314,domain_households=5748,frame_n=8152,households=6217,zero_domain_households=469,strata=9,df=6208,tie_changed_households=3,other_columns_unchanged=True))
    base=next(a for a in read(WORK/'work/elsa-mnlfa-precision-2026-10-24/results/model_summaries.json') if a['model']=='slope_adjusted')
    releases={a['model']:a for a in read(WORK/'work/elsa-mnlfa-inference-2026-10-25/results_anchors/model_summaries.json')}
    jobs=[]
    for name,anch,items,source in [('AS01_S4',[0,1,2,3],[4,5,6,7],base)]+[(r['id'],[int(i[1:])-1 for i in r['remaining_anchors']],[4,5,6,7,int(r['released'][1:])-1],releases[r['id']]) for r in plan['models']['releases']]:
        m=Model('slope_adjusted',81,anchors=anch)
        if name!='AS01_S4':assert source['numerically_admissible'] and source['fit_q']==61
        theta=np.array([source['parameter_values'][k] for k in m.names]);assert len(theta)==(54 if name=='AS01_S4' else 60)
        jobs.append(dict(id=name,anchors=anch,items=items,theta=theta,parameter_names=m.names))
    # Point estimates are known before new covariance calculations; preserve all 36 regardless of gates.
    point={}
    for job in jobs:
        m=Model('slope_adjusted',81,anchors=job['anchors']);values,J=inputs(m,job['theta'],x,w,job['items'])
        point[job['id']]={('i'+str(j+1)):float(v) for j,v in zip(job['items'],normalized_mean_influence(values,x[:,0],w)['mean'])}
    point_rows=[dict(**t,estimate_pp=100*(point[t['model']][t['item']]-(point['AS01_S4'][t['item']] if t['kind']=='paired_change' else 0))) for t in targets]
    pd.DataFrame(point_rows).to_csv(OUT/'fixed_points.csv',index=False)
    names=[j['id'] for j in jobs];all_models={};model_checks={}
    progress('running',models=names,completed_models=[],workers=5)
    with ProcessPoolExecutor(max_workers=5) as pool:
        futures={pool.submit(run_model,j,y,x,w,wn):j['id'] for j in jobs}
        for f in as_completed(futures):
            try:name,models,gates=f.result()
            except (FloatingPointError,np.linalg.LinAlgError) as exc:name,models,gates=worker_failure(futures[f],exc)
            all_models[name]=models;model_checks[name]=gates
            progress('running',models=names,completed_models=[n for n in names if n in all_models],workers=5)
    bundles={};tie_bundles={}
    for v in VARIANTS:
        models={name:all_models[name][v] for name in names}
        bundles[v]=assemble(models,targets,hi,strata);tie_bundles[v]=assemble(models,targets,hi,tie)
        if bundles[v] is not None:assert bundles[v]['df']==6208
        dump(OUT/('joint_'+v+'.json'),dict(primary=bundles[v],tie=tie_bundles[v],diagnostic_only=True,intervals_released=False))
    checks={};diagnostics={}
    for t in targets:checks[t['id']],diagnostics[t['id']]=target_gate(bundles,t['id'])
    allowed,reasons=release_decisions(model_checks,checks,targets)
    tie_checks={};tie_diagnostics={}
    for t in targets:tie_checks[t['id']],tie_diagnostics[t['id']]=target_gate(tie_bundles,t['id'])
    tie_allowed,tie_reasons=release_decisions(model_checks,tie_checks,targets)
    for ident in tie_allowed:
        if not allowed[ident]:tie_allowed[ident]=False;tie_reasons[ident]=['primary_not_released']+reasons[ident]+tie_reasons[ident]
    verify()
    critical=float(student_t.ppf(.975,6208));bonf=float(student_t.ppf(1-.05/72,6208))
    rows=[]
    for t,pointrow in zip(targets,point_rows):
        ident=t['id'];row=dict(**t,estimate_pp=pointrow['estimate_pp'],intervals_released=allowed[ident],failure_reason='; '.join(reasons[ident]),tie_intervals_released=tie_allowed[ident],tie_failure_reason='; '.join(tie_reasons[ident]),family_size=36,df=6208)
        for field in ['se_pp','lower_pp','upper_pp','bonferroni_lower_pp','bonferroni_upper_pp','tie_se_pp','tie_lower_pp','tie_upper_pp','tie_bonferroni_lower_pp','tie_bonferroni_upper_pp']:row[field]=None
        if allowed[ident]:
            b=bundles['q81'];tb=tie_bundles['q81'];j=b['target_ids'].index(ident);jt=tb['target_ids'].index(ident)
            assert abs(100*b['estimates'][j]-row['estimate_pp'])<1e-10
            assert abs(tb['estimates'][jt]-b['estimates'][j])<1e-14
            se=100*b['standard_errors'][j];estimate=row['estimate_pp']
            row.update(se_pp=se,lower_pp=estimate-critical*se,upper_pp=estimate+critical*se,bonferroni_lower_pp=estimate-bonf*se,bonferroni_upper_pp=estimate+bonf*se)
            if tie_allowed[ident]:
                ts=100*tb['standard_errors'][jt]
                row.update(tie_se_pp=ts,tie_lower_pp=estimate-critical*ts,tie_upper_pp=estimate+critical*ts,tie_bonferroni_lower_pp=estimate-bonf*ts,tie_bonferroni_upper_pp=estimate+bonf*ts)
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT/'intervals.csv',index=False)
    dump(OUT/'release_gates.json',dict(model_checks=model_checks,target_checks=checks,target_diagnostics=diagnostics,release_flags=allowed,failure_reasons=reasons,tie_target_checks=tie_checks,tie_target_diagnostics=tie_diagnostics,tie_release_flags=tie_allowed,tie_failure_reasons=tie_reasons))
    verify()
    completion=dict(status='completed',completed_utc=stamp(),seconds=time.time()-START,models=5,direct_targets=20,paired_targets=16,intervals_released=sum(allowed.values()),tie_intervals_released=sum(tie_allowed.values()),family_size=36,df=6208,
                    pointwise_critical=critical,bonferroni_critical=bonf,freeze_sha256=sha(ROOT/'freeze.json'),python=sys.version,numpy=np.__version__,scipy=scipy.__version__,pandas=pd.__version__,
                    participant_rows_exported=0,influence_arrays_exported=0,optimization_calls=0,output_hashes={p.name:sha(p) for p in OUT.glob('*') if p.is_file() and not p.name.endswith('progress.json')})
    dump(OUT/'completion.json',completion);progress('completed',intervals_released=sum(allowed.values()),total_targets=36)
if __name__=='__main__':
    try:main()
    except Exception as exc:
        if RUN_STARTED:
            if (OUT/'fixed_points.csv').exists():
                ledger=pd.read_csv(OUT/'fixed_points.csv');ledger['intervals_released']=False;ledger['tie_intervals_released']=False
                ledger['failure_reason']='run_failed_'+type(exc).__name__;ledger['family_size']=36;ledger['df']=6208
                for col in ['se_pp','lower_pp','upper_pp','bonferroni_lower_pp','bonferroni_upper_pp']:ledger[col]=None
                ledger.to_csv(OUT/'failed_run_ledger.csv',index=False)
            progress('failed',error_type=type(exc).__name__,result_validity='NO_SCIENTIFIC_INTERVALS_RELEASED')
        raise
