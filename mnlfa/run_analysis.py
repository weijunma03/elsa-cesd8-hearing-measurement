"""Frozen, bounded exploratory pilot. Aggregate outputs only; no participant export."""
from pathlib import Path
import sys, json, hashlib, time, datetime, os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize
from scipy.integrate import quad_vec
from scipy.special import ndtr
from engine import Model, STAGES, NONANCHORS
from data_audit.audit_covariates import load_covariate_frame
from engine import group_rows

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'
PLAN=json.loads((ROOT/'analysis_plan.json').read_text(encoding='utf-8'))
CFG=PLAN['estimation']
START=time.time()
ATTEMPTS=[]
def stamp(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(path,obj):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    tmp.replace(path)
def progress(status,**kw):
    d=dict(status=status,updated_utc=stamp(),elapsed_seconds=time.time()-START,**kw)
    dump(OUT/'progress.json',d)
    print(json.dumps(d,ensure_ascii=False),flush=True)
def guard(event,args):
    if event.startswith('socket.'): raise RuntimeError('Network disabled for local analysis')
sys.addaudithook(guard)

def projected(g,t,bounds):
    g=g.copy()
    for j,(lo,hi) in enumerate(bounds):
        if (t[j]<=lo+1e-8 and g[j]>0) or (t[j]>=hi-1e-8 and g[j]<0): g[j]=0
    return float(np.max(np.abs(g)))
def targets(m,t,profiles,weights):
    delta=[]
    for age,female in profiles:
        delta.append(m.common_reference(t,[1,age,female])-m.common_reference(t,[0,age,female]))
    return np.average(np.asarray(delta),axis=0,weights=weights)
def perturb(t,m):
    bounds=np.asarray(m.bounds)
    return np.clip(t+.08*np.sin(np.arange(1,m.p+1)),bounds[:,0]+.001,bounds[:,1]-.001)

def fit_stage(stage,groups,initial,profiles,weights):
    chosen=None
    resolutions=[]
    t0=initial.copy()
    previous_target=None
    for q,qcheck in ((21,31),(31,41),(41,61)):
        m=Model(stage,q)
        starts=[t0,perturb(t0,m)]
        fits=[]
        for sid,tstart in enumerate(starts):
            before=time.time()
            progress('fitting',model=stage,quadrature=q,start=sid+1)
            iteration=[0]
            def callback(xk):
                iteration[0]+=1
                if iteration[0]%25==0:
                    progress('fitting',model=stage,quadrature=q,start=sid+1,iteration=iteration[0])
            try:
                opt=minimize(m.fg,tstart,args=(groups,),jac=True,method='L-BFGS-B',bounds=m.bounds,
                    options=dict(maxiter=CFG['optimizer_maxiter'],ftol=CFG['optimizer_ftol'],gtol=CFG['optimizer_gtol'],maxls=40),callback=callback)
                value,grad=m.fg(opt.x,groups)
                record=dict(model=stage,q=q,start=sid+1,success=bool(opt.success),message=str(opt.message),
                    iterations=int(opt.nit),evaluations=int(opt.nfev),nll=value,gradient_max=float(np.max(np.abs(grad))),
                    projected_gradient=projected(grad,opt.x,m.bounds),bound_hits=m.bounds_hits(opt.x),
                    seconds=time.time()-before,parameters=opt.x.tolist())
                fits.append((value,opt.x.copy(),record))
            except Exception as exc:
                record=dict(model=stage,q=q,start=sid+1,success=False,error=repr(exc),seconds=time.time()-before)
            ATTEMPTS.append(record)
            dump(OUT/'attempts.json',ATTEMPTS)
            progress('start_finished',**{k:v for k,v in record.items() if k not in ('parameters','bound_hits')})
        if not fits: raise RuntimeError('Both starts failed at '+stage+' q'+str(q))
        fits.sort(key=lambda v:v[0])
        value,t,rec=fits[0]
        mc=Model(stage,qcheck)
        vcheck,gcheck=mc.fg(t,groups)
        target=targets(m,t,profiles,weights)
        gap=None if len(fits)<2 else float(max(v[0] for v in fits)-min(v[0] for v in fits))
        resolution=dict(q=q,check_q=qcheck,selected_start=rec['start'],nll=value,check_nll=vcheck,
            check_nll_difference=abs(value-vcheck),check_gradient_max=float(np.max(np.abs(gcheck))),
            multistart_gap=gap,selected_optimizer_success=rec['success'],projected_gradient=rec['projected_gradient'],
            target=target.tolist(),target_change_from_previous_q=None if previous_target is None else (target-previous_target).tolist())
        resolution['quadrature_pass']=bool(abs(value-vcheck)<=CFG['quadrature_mean_nll_tolerance'] and np.max(np.abs(gcheck))<=CFG['quadrature_gradient_max_tolerance'])
        resolution['multistart_pass']=bool(len(fits)==2 and all(f[2]['success'] and f[2]['projected_gradient']<=CFG['projected_gradient_acceptance'] for f in fits) and gap<=CFG['multistart_mean_objective_tolerance'])
        resolutions.append(resolution)
        chosen=(m,mc,t,rec,resolution)
        dump(OUT/(stage+'_resolutions.json'),resolutions)
        if resolution['quadrature_pass']: break
        t0=t.copy()
        previous_target=target.copy()
    m,mc,t,rec,res=chosen
    progress('hessian_check',model=stage,quadrature=mc.q)
    hess=np.empty((m.p,m.p))
    for j in range(m.p):
        step=1e-4*max(1.,abs(t[j])); hi=t.copy(); lo=t.copy(); hi[j]+=step; lo[j]-=step
        hess[:,j]=(mc.fg(hi,groups)[1]-mc.fg(lo,groups)[1])/(2*step)
    asym=float(np.max(np.abs(hess-hess.T)))
    hess=(hess+hess.T)/2
    eigen=np.linalg.eigvalsh(hess)
    hits=m.bounds_hits(t)
    integration_error=0.
    for age,female in profiles:
        for h in (0,1):
            x=np.array([h,age,female]); c=m.to_components(t)
            b=c['b']+c['delta']@x; a=np.exp(c['loga']+c['omega']@x)
            numeric,_=quad_vec(lambda z:ndtr(b+a*z)*np.exp(-z*z/2)/np.sqrt(2*np.pi),-10,10,epsabs=1e-10,epsrel=1e-10)
            integration_error=max(integration_error,float(np.max(np.abs(numeric-m.common_reference(t,x)))))
    checks=dict(optimizer=bool(rec['success'] and rec['projected_gradient']<=CFG['projected_gradient_acceptance']),
        multistart=res['multistart_pass'],quadrature=res['quadrature_pass'],no_bound_hits=not hits,
        positive_observed_hessian=bool(eigen[0]>1e-7),reference_integration=bool(integration_error<=CFG['quadrature_reference_probability_tolerance']))
    summary=dict(model=stage,parameters=m.p,fit_q=m.q,check_q=mc.q,nll=rec['nll'],checks=checks,
        numerically_admissible=all(checks.values()),bound_hits=hits,
        hessian_min_eigenvalue=float(eigen[0]),hessian_max_eigenvalue=float(eigen[-1]),
        hessian_condition_number=float(eigen[-1]/eigen[0]) if eigen[0]>0 else None,
        hessian_asymmetry_max=asym,hessian_eigenvalues=eigen.tolist(),reference_integration_error=integration_error,
        check_gradient_max=res['check_gradient_max'],check_nll_difference=res['check_nll_difference'],multistart_gap=res['multistart_gap'],
        resolutions=resolutions,parameter_values=dict(zip(m.names,t.tolist())))
    dump(OUT/(stage+'_fit.json'),summary)
    progress('model_finished',model=stage,numerically_admissible=summary['numerically_admissible'],checks=checks)
    return m,mc,t,summary

def main():
    if OUT.exists() and (OUT/'progress.json').exists(): raise RuntimeError('Run already exists; no silent overwrite/retry')
    freeze=json.loads((ROOT/'freeze.json').read_text(encoding='utf-8'))
    for file,digest in freeze['files'].items():
        if sha(ROOT/file)!=digest: raise RuntimeError('Frozen source mismatch '+file)
    OUT.mkdir(exist_ok=True)
    progress('loading',freeze_sha256=sha(ROOT/'freeze.json'))
    f,_=load_covariate_frame('lex_first')
    d=f.loc[f.domain.eq(1)].copy()
    assert len(d)==7314 and d.hh.nunique()==5748
    x=d[['g','age10','female']].to_numpy(float); y=d[[f'i{j}' for j in range(1,9)]].to_numpy(float)
    assert np.isfinite(x).all() and np.isin(y,[0,1]).all()
    w=d.w.to_numpy(dtype=float,copy=True)
    for h in (0,1):
        ix=x[:,0]==h; assert ix.sum()==(5241 if h==0 else 2073)
        w[ix]*=ix.sum()/w[ix].sum()
    profiles,pi=np.unique(x[:,1:],axis=0,return_inverse=True)
    pw=np.bincount(pi,weights=w)
    grouped=group_rows(y,x,w)
    x0=x.copy(); x0[:,1:]=0
    g0=group_rows(y,x0,w) # exact lossless collapse of unused age/sex in hearing-only stage
    summaries=[]; contrasts=[]; params=[]; local=[]; prows=[]; previous=None
    for stage in STAGES:
        m=Model(stage,21); groups=g0 if stage=='hearing_only' else grouped
        init=m.initial(groups) if previous is None else m.embed(previous[1],previous[0])
        m,mc,t,summary=fit_stage(stage,groups,init,profiles,pw)
        previous=(m,t)
        summaries.append(summary)
        for name,value in zip(m.names,t): params.append(dict(model=stage,parameter=name,value=value))
        target=targets(m,t,profiles,pw)
        for j in NONANCHORS:
            contrasts.append(dict(model=stage,item='i'+str(j+1),contrast=target[j] if summary['numerically_admissible'] else None,contrast_pp=100*target[j] if summary['numerically_admissible'] else None,numerically_admissible=summary['numerically_admissible']))
        for age in (60,70,80):
            for female in (0,1):
                probs=[m.common_reference(t,[h,(age-70)/10,female]) for h in (0,1)]
                counts=[int(((d.g==h)&(d.female==female)&(d.age_capped90==age)).sum()) for h in (0,1)]
                bands=[int(((d.g==h)&(d.female==female)&(abs(d.age_capped90-age)<=2)).sum()) for h in (0,1)]
                for j in NONANCHORS:
                    prows.append(dict(model=stage,item='i'+str(j+1),age=age,female=female,
                        reference_p0=probs[0][j] if summary['numerically_admissible'] else None,reference_p1=probs[1][j] if summary['numerically_admissible'] else None,contrast_pp=100*(probs[1][j]-probs[0][j]) if summary['numerically_admissible'] else None,
                        n0_exact_age=counts[0],n1_exact_age=counts[1],n0_age_plusminus2=bands[0],n1_age_plusminus2=bands[1],numerically_admissible=summary['numerically_admissible']))
        for h in (0,1):
            ix=x[:,0]==h; wh=w[ix]; yh=y[ix]; xx=x[ix]
            obsm=np.average(yh,axis=0,weights=wh); obsp=(yh*wh[:,None]).T@yh/wh.sum()
            xu,inv=np.unique(xx,axis=0,return_inverse=True); wx=np.bincount(inv,weights=wh)
            predm=np.zeros(8); predp=np.zeros((8,8))
            for row,weight in zip(xu,wx):
                mm,pp=mc.moments(t,row); predm+=weight*mm/wh.sum(); predp+=weight*pp/wh.sum()
            for j in range(8):
                local.append(dict(model=stage,group='O'+str(h),kind='item',item1='i'+str(j+1),item2='',observed=obsm[j],predicted=predm[j],residual=obsm[j]-predm[j],numerically_admissible=summary['numerically_admissible']))
                for k in range(j+1,8):
                    local.append(dict(model=stage,group='O'+str(h),kind='pair',item1='i'+str(j+1),item2='i'+str(k+1),observed=obsp[j,k],predicted=predp[j,k],residual=obsp[j,k]-predp[j,k],numerically_admissible=summary['numerically_admissible']))
        pd.DataFrame(contrasts).to_csv(OUT/'reference_contrasts.csv',index=False)
        pd.DataFrame(params).to_csv(OUT/'parameters.csv',index=False)
        pd.DataFrame(prows).to_csv(OUT/'profiles.csv',index=False)
        pd.DataFrame(local).to_csv(OUT/'local_fit.csv',index=False)
        dump(OUT/'model_summaries.json',summaries)
    protected=json.loads((ROOT/'data_audit/audit.json').read_text(encoding='utf-8'))
    completion=dict(completed_utc=stamp(),seconds=time.time()-START,n=7314,households=5748,groups={'O0':5241,'O1':2073},
        four_models_complete=len(summaries)==4,numerically_admissible=sum(s['numerically_admissible'] for s in summaries),
        freeze_sha256=sha(ROOT/'freeze.json'),python=sys.version,numpy=np.__version__,scipy=scipy.__version__,pandas=pd.__version__,
        inference='Exploratory point estimates only. No design-based CI, p values, causal or confirmed-DIF claims.',
        output_hashes={p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='progress.json'})
    dump(OUT/'completion.json',completion)
    progress('completed',models=len(summaries),numerically_admissible=completion['numerically_admissible'])

if __name__=='__main__':
    try: main()
    except Exception as exc:
        if OUT.exists(): progress('failed',error=repr(exc))
        raise
