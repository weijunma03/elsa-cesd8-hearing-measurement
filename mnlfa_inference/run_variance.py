"""Frozen working-design MNLFA inference. Only aggregate outputs are saved."""
from pathlib import Path
import os, sys, json, hashlib, datetime, time, importlib.util
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key] = '1'
import numpy as np
import pandas as pd
import scipy
from scipy.special import ndtr
from scipy.stats import t as student_t
from engine import Model, group_rows, STAGES, NONANCHORS, X_NAMES
from variance import normalized_mean_influence, household_covariance, sandwich_targets

ROOT=Path(__file__).resolve().parent
WORKSPACE=ROOT.parents[1]
OUT=ROOT/'results_variance'
PLAN=json.loads((ROOT/'analysis_plan.json').read_text(encoding='utf-8'))
CFG=PLAN['variance']
START=time.time()
RUN_STARTED=False

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def stamp(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def dump(path,obj):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    tmp.replace(path)
def progress(status,**kwargs):
    record=dict(status=status,updated_utc=stamp(),seconds=time.time()-START,**kwargs)
    dump(OUT/'progress.json',record)
    print(json.dumps(record,ensure_ascii=False),flush=True)
def guard(event,args):
    if event.startswith('socket.'): raise RuntimeError('Network disabled for local analysis')
sys.addaudithook(guard)

def verify_freeze():
    freeze=json.loads((ROOT/'freeze.json').read_text(encoding='utf-8'))
    for rel,digest in freeze['files'].items():
        if sha(ROOT/rel)!=digest: raise RuntimeError('Frozen code changed: '+rel)
    for path,digest in freeze['protected'].items():
        if sha(path)!=digest: raise RuntimeError('Protected prior file changed: '+path)
    return freeze

def reference(m,theta,xx,items=NONANCHORS):
    """Vectorized standard-normal reference probabilities and analytic Jacobian."""
    xx=np.atleast_2d(np.asarray(xx,float)); items=np.asarray(items,int)
    c=m.to_components(theta)
    b=c['b'][None,:]+xx@c['delta'].T
    a2=np.exp(2*(c['loga'][None,:]+xx@c['omega'].T))
    den=np.sqrt(1+a2); z=b/den
    pdf=np.exp(-z*z/2)/np.sqrt(2*np.pi)
    db=pdf/den; dl=-pdf*b*a2/(den**3)
    jac=np.zeros((len(xx),len(items),m.p))
    lookup={name:k for k,name in enumerate(m.names)}
    for l,j in enumerate(items):
        jac[:,l,lookup[f'b_i{j+1}']]=db[:,j]
        jac[:,l,lookup[f'loga_i{j+1}']]=dl[:,j]
        for k,name in enumerate(X_NAMES):
            bn=f'intercept_i{j+1}_{name}'; an=f'logslope_i{j+1}_{name}'
            if bn in lookup: jac[:,l,lookup[bn]]=db[:,j]*xx[:,k]
            if an in lookup: jac[:,l,lookup[an]]=dl[:,j]*xx[:,k]
    return ndtr(z)[:,items],jac

def contrasts(m,theta,age_sex):
    age_sex=np.atleast_2d(age_sex)
    p0,j0=reference(m,theta,np.column_stack((np.zeros(len(age_sex)),age_sex)))
    p1,j1=reference(m,theta,np.column_stack((np.ones(len(age_sex)),age_sex)))
    return p1-p0,j1-j0

def target_inputs(m,theta,x,wn):
    profiles,inv=np.unique(x[:,1:],axis=0,return_inverse=True)
    pw=np.bincount(inv,weights=wn)
    v,j=contrasts(m,theta,profiles)
    meanj=np.einsum('n,nkp->kp',pw,j)/pw.sum()
    fixed_profiles=np.array([[(age-70)/10,female] for age in (60,70,80) for female in (0,1)],float)
    fv,fj=contrasts(m,theta,fixed_profiles)
    meta=[dict(kind='average',item='i'+str(j+1),age=None,female=None) for j in NONANCHORS]
    for age in (60,70,80):
        for female in (0,1):
            meta.extend(dict(kind='profile',item='i'+str(j+1),age=age,female=female) for j in NONANCHORS)
    return v[inv],meanj,fv.ravel(),fj.reshape(-1,m.p),meta

def observed_hessian(m,theta,grouped,notify=True):
    A=np.empty((m.p,m.p))
    for j in range(m.p):
        h=CFG['hessian_step']*max(1.,abs(theta[j]))
        hi=theta.copy(); lo=theta.copy(); hi[j]+=h; lo[j]-=h
        A[:,j]=(m.fg(hi,grouped)[1]-m.fg(lo,grouped)[1])/(2*h)
        if notify and (j+1)%10==0: progress('hessian',model=m.stage,q=m.q,column=j+1,total=m.p)
    asym=float(np.max(np.abs(A-A.T)))
    return (A+A.T)/2,asym

def run_q(stage,q,theta,grouped,inverse,xx,raw_w,household_index,stratum_first,stratum_last,target):
    m=Model(stage,q); vals,Jac,fixed,FJac,meta=target
    progress('scores',model=stage,q=q)
    nll,grad=m.fg(theta,grouped)
    ps=m.loglik_scores(theta,grouped)
    assert np.all(inverse>=0)
    assert np.array_equal(ps.x[inverse],xx)
    scores=-ps.scores[inverse]
    score_mean=normalized_mean_influence(scores,xx[:,0],raw_w)['mean']
    score_error=float(np.max(np.abs(grad-score_mean)))
    if score_error>1e-10: raise AssertionError('Pattern score mapping/normalization error')
    A,asym=observed_hessian(m,theta,grouped)
    eig=np.linalg.eigvalsh(A)
    detail=dict(q=q,nll=float(nll),gradient=grad.tolist(),score_mean=score_mean.tolist(),gradient_max=float(np.max(np.abs(grad))),
        diagnostic_only=True,intervals_not_released=True,
        release_policy='Raw quadrature diagnostics only. Scientific intervals are released exclusively in intervals.csv after all model-level gates pass.',
        parameter_sha256=hashlib.sha256(theta.astype('<f8').tobytes()).hexdigest(),
        score_identity_error=score_error,hessian=A.tolist(),hessian_asymmetry=asym,
        hessian_eigenvalues=eig.tolist(),hessian_min_eigenvalue=float(eig[0]))
    if eig[0]<=0:
        detail['covariance_available']=False
        return detail,None
    result=sandwich_targets(scores,xx[:,0],raw_w,A,vals,Jac,household_index,stratum_first,fixed,FJac)
    tie=household_covariance(result['target_row_influence'],household_index,stratum_last)
    tie_se=np.sqrt(np.diag(tie['covariance']))
    all_j=np.vstack((Jac,FJac))
    newton=-np.linalg.solve(A,grad)
    target_correction=all_j@newton
    keep=('estimates','covariance','standard_errors','intervals','theta_covariance','direct_covariance',
          'parameter_covariance','direct_parameter_cross_covariance')
    detail.update({key:result[key].tolist() for key in keep})
    detail.update(covariance_available=True,df=int(result['df']),conditioning=result['conditioning'],
        covariance_decomposition_error=result['maximum_covariance_decomposition_error'],
        newton_target_correction=target_correction.tolist(),newton_parameter_max=float(np.max(np.abs(newton))),
        target_jacobian=all_j.tolist(),tie_standard_errors=tie_se.tolist(),tie_covariance=tie['covariance'].tolist(),
        household_design={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in result['household_design'].items()})
    return detail,result

def main():
    global RUN_STARTED
    if (OUT/'progress.json').exists(): raise RuntimeError('Run exists; refuse silent retry')
    verify_freeze(); OUT.mkdir(exist_ok=True)
    RUN_STARTED=True
    progress('loading',freeze_sha256=sha(ROOT/'freeze.json'))
    spec=importlib.util.spec_from_file_location('frozen_covariate_loader',WORKSPACE/'work/elsa-mnlfa-precision-2026-10-24/data_audit/audit_covariates.py')
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    frame,audit=mod.load_covariate_frame('lex_first')
    alternate,_=mod.load_covariate_frame('lex_last')
    assert frame.drop(columns='stratum').equals(alternate.drop(columns='stratum'))
    d=frame.loc[frame.domain.eq(1)].copy()
    assert len(frame)==8152 and len(d)==7314 and d.hh.nunique()==5748
    hh=frame[['hh','stratum']].drop_duplicates('hh').sort_values('hh').reset_index(drop=True)
    hh2=alternate[['hh','stratum']].drop_duplicates('hh').sort_values('hh').reset_index(drop=True)
    assert frame.groupby('hh').stratum.nunique().max()==1
    assert alternate.groupby('hh').stratum.nunique().max()==1
    assert np.array_equal(hh.hh,hh2.hh) and len(hh)==6217 and hh.stratum.nunique()==9
    assert (hh.stratum!=hh2.stratum).sum()==3
    dump(OUT/'design_audit.json',dict(full_frame_n=len(frame),domain_n=len(d),full_households=len(hh),
        domain_households=int(d.hh.nunique()),zero_domain_households=int(len(hh)-d.hh.nunique()),
        strata=int(hh.stratum.nunique()),df=int(len(hh)-hh.stratum.nunique()),
        tie_changed_households=int((hh.stratum!=hh2.stratum).sum()),
        tie_changed_person_rows=int((frame.stratum!=alternate.stratum).sum()),
        other_frame_columns_identical=True,prior_summaries_sha256=sha(WORKSPACE/'work/elsa-mnlfa-precision-2026-10-24/results/model_summaries.json')))
    hi=pd.Index(hh.hh).get_indexer(d.hh)
    ss=hh.stratum.to_numpy(); ss2=hh2.stratum.to_numpy()
    x=d[['g','age10','female']].to_numpy(float)
    y=d[[f'i{j}' for j in range(1,9)]].to_numpy(float)
    raw_w=d.w.to_numpy(dtype=float,copy=True); wn=raw_w.copy()
    for h in (0,1):
        ix=x[:,0]==h
        assert ix.sum()==(5241 if h==0 else 2073)
        wn[ix]*=ix.sum()/wn[ix].sum()
    prior=json.loads((WORKSPACE/'work/elsa-mnlfa-precision-2026-10-24/results/model_summaries.json').read_text(encoding='utf-8'))
    prior={s['model']:s for s in prior}
    summaries=[]; rows=[]
    for stage in STAGES:
        m=Model(stage,CFG['primary_q'])
        theta=np.asarray([prior[stage]['parameter_values'][n] for n in m.names])
        target=target_inputs(m,theta,x,wn)
        xx=x.copy()
        if stage=='hearing_only': xx[:,1:]=0
        grouped,inverse=group_rows(y,xx,wn,return_inverse=True)
        results=[]
        for q in (CFG['primary_q'],CFG['check_q']):
            detail,result=run_q(stage,q,theta,grouped,inverse,xx,raw_w,hi,ss,ss2,target)
            dump(OUT/f'{stage}_q{q}.json',detail)
            results.append(detail)
            del result
        a,b=results
        available=a['covariance_available'] and b['covariance_available']
        rel=None; newton_ok=False
        if available:
            se=np.asarray(a['standard_errors']); se2=np.asarray(b['standard_errors'])
            rel=np.abs(se-se2)/np.maximum(se2,CFG['se_relative_denominator_floor'])
            newton_ok=all(np.all(np.abs(r['newton_target_correction'])<=np.maximum(CFG['newton_target_absolute_floor'],CFG['newton_target_fraction_se']*np.asarray(r['standard_errors']))) for r in results)
        checks=dict(both_stationary=all(r['gradient_max']<=CFG['max_gradient'] for r in results),
            both_hessians_positive=all(r['hessian_min_eigenvalue']>CFG['hessian_min_eigenvalue'] for r in results),
            all_target_se_stable=bool(available and np.max(rel)<=CFG['max_se_relative_difference']),
            newton_target_correction_small=bool(newton_ok))
        valid=all(checks.values())
        summary=dict(model=stage,parameters=m.p,checks=checks,intervals_released=valid,
            primary_q=CFG['primary_q'],check_q=CFG['check_q'],max_se_relative_difference=None if rel is None else float(np.max(rel)),
            max_mean_se_relative_difference=None if rel is None else float(np.max(rel[:4])),
            parameter_values=prior[stage]['parameter_values'],point_estimates_unchanged=True)
        if available:
            summary.update(max_tie_se_relative_difference=float(np.max(np.abs(np.asarray(a['tie_standard_errors'])-se)/np.maximum(se,1e-8))),
                df=a['df'],household_design=a['household_design'])
        for k,meta in enumerate(target[-1]):
            row=dict(model=stage,**meta,intervals_released=valid)
            # Exact closed-form point estimates remain reportable from the previous accepted fits.
            estimate=float(normalized_mean_influence(target[0],x[:,0],raw_w)['mean'][k]) if k<4 else float(target[2][k-4])
            row['estimate_pp']=100*estimate
            for key in ('se_pp','lower_pp','upper_pp','bonferroni_lower_pp','bonferroni_upper_pp','parameter_only_se_pp','tie_se_pp'):
                row[key]=None
            if valid:
                row.update(se_pp=100*se[k],lower_pp=100*a['intervals'][k][0],upper_pp=100*a['intervals'][k][1],
                    parameter_only_se_pp=100*np.sqrt(a['parameter_covariance'][k][k]),tie_se_pp=100*a['tie_standard_errors'][k])
                if k<4:
                    critical=float(student_t.ppf(1-.05/(2*16),a['df']))
                    row.update(bonferroni_lower_pp=100*(estimate-critical*se[k]),bonferroni_upper_pp=100*(estimate+critical*se[k]))
            rows.append(row)
        summaries.append(summary)
        pd.DataFrame(rows).to_csv(OUT/'intervals.csv',index=False)
        dump(OUT/'model_summaries.json',summaries)
        progress('model_finished',model=stage,checks=checks,intervals_released=valid,max_se_relative_difference=summary['max_se_relative_difference'])
    verify_freeze()
    completion=dict(completed_utc=stamp(),seconds=time.time()-START,models=len(summaries),
        models_with_intervals=sum(s['intervals_released'] for s in summaries),n=7314,full_households=6217,df=6208,
        average_targets=16,profile_targets=96,family_size=16,freeze_sha256=sha(ROOT/'freeze.json'),
        python=sys.version,numpy=np.__version__,scipy=scipy.__version__,participant_rows_exported=0,
        output_hashes={p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='progress.json'})
    dump(OUT/'completion.json',completion)
    progress('completed',models=4,models_with_intervals=completion['models_with_intervals'])

def synthetic_checks():
    rng=np.random.default_rng(711)
    checks=[]; maxerr=0.
    for stage in STAGES:
        m=Model(stage,9); theta=rng.normal(0,.08,m.p)
        xx=np.array([[0,-1,0],[1,1,1],[1,0,0],[0,2,1]],float)
        v,jac=reference(m,theta,xx)
        checks.append(bool(np.allclose(v,np.array([m.common_reference(theta,row)[NONANCHORS] for row in xx]),atol=1e-14)))
        for p in range(m.p):
            hi=theta.copy(); lo=theta.copy(); hi[p]+=1e-6; lo[p]-=1e-6
            fd=(reference(m,hi,xx)[0]-reference(m,lo,xx)[0])/2e-6
            err=float(np.max(np.abs(fd-jac[:,:,p]))); maxerr=max(err,maxerr)
            checks.append(err<1e-8)
        v,j=contrasts(m,theta,xx[:,1:]); checks.append(v.shape==(4,4) and j.shape==(4,4,m.p))
    record=dict(passed=all(checks),checks=len(checks),max_jacobian_error=maxerr,real_data_read=False)
    dump(ROOT/'variance_runner_synthetic_checks.json',record)
    print(json.dumps(record))
    assert all(checks)

if __name__=='__main__':
    if '--test' in sys.argv: synthetic_checks()
    else:
        try: main()
        except Exception as exc:
            if RUN_STARTED: progress('failed',error=repr(exc))
            raise
