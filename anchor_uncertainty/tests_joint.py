"""Independent exact-root and multiplier tests; synthetic input only."""
import os,sys
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib,copy,datetime
import numpy as np
from engine import Model
from joint import reference,inputs,linearize,assemble,model_gate,target_gate,release_decisions,VARIANTS
ROOT=Path(__file__).resolve().parent
plan=json.loads((ROOT/'analysis_plan.json').read_text(encoding='utf-8-sig'));targets=plan['estimands']['targets']
checks=[]
def check(name,value,detail=None):checks.append(dict(check=name,passed=bool(value),detail=detail))
def near(name,a,b,tol=2e-9):
    err=float(np.max(np.abs(np.asarray(a)-np.asarray(b))));check(name,np.isfinite(err) and err<=tol,dict(max_error=err,tolerance=tol))
rng=np.random.default_rng(2026102502)
g=np.array([0,1,0,0,1,1,0,1,1,0,1,1]);w=np.array([.4,1.4,2.3,.7,2.4,3.,.6,1.7,.8,2.,1.9,.3]);hi=np.array([0,0,1,1,1,3,3,4,4,6,6,7]);ss=np.repeat(np.arange(3),3)
n=12;H=9;p=3;ones=np.ones(H)
def mean(v,m):
    cm=m[hi];result=np.zeros(v.shape[1:])
    for group in (0,1):
        ix=g==group;weights=cm[ix]*w[ix]
        result+=cm[ix].sum()/cm.sum()*np.einsum('n,n...->...',weights,v[ix])/weights.sum()
    return result
A=np.array([[1.4,.3,-.1],[.3,1.1,.2],[-.1,.2,1.8]])
b0=rng.normal(size=(n,p));v0=rng.normal(size=(n,5));jr0=rng.normal(0,.3,size=(n,5,p))
models={};exact={}
for k,name in enumerate(['AS01_S4']+[r['id'] for r in plan['models']['releases']]):
    count=4 if k==0 else 5;items=['i5','i6','i7','i8']+([] if k==0 else [plan['models']['releases'][k-1]['released']])
    Ak=A+np.eye(p)*k*.06;b=b0+k*.015*rng.normal(size=(n,p));v=v0[:,:count]+k*.02*rng.normal(size=(n,count));jr=jr0[:,:count]+k*.01*rng.normal(size=(n,count,p))
    theta=-np.linalg.solve(Ak,mean(b,ones));scores=b+theta@Ak.T;values=v+np.einsum('nkp,p->nk',jr,theta);J=mean(jr,ones)
    models[name]=linearize(scores,g,w,Ak,values,J);models[name]['items']=items
    exact[name]=(Ak,b,v,jr)
def exact_all(m):
    values={}
    for name,(a,b,v,jr) in exact.items():
        theta=-np.linalg.solve(a,mean(b,m));values[name]=mean(v+np.einsum('nkp,p->nk',jr,theta),m)
    return np.array([values[t['model']][models[t['model']]['items'].index(t['item'])]-(values['AS01_S4'][models['AS01_S4']['items'].index(t['item'])] if t['kind']=='paired_change' else 0) for t in targets])
def fd(func):
    result=[]
    for h in range(H):
        up=ones.copy();down=ones.copy();up[h]+=2e-5;down[h]-=2e-5;result.append((func(up)-func(down))/4e-5)
    return np.asarray(result)
def directcov(hh):
    V=np.zeros((hh.shape[1],hh.shape[1]))
    for s in np.unique(ss):
        part=hh[ss==s];m=sum(part)/len(part)
        for v in part:V+=len(part)/(len(part)-1)*np.outer(v-m,v-m)
    return V
bundle=assemble(models,targets,hi,ss);F=fd(exact_all)
near('all36 exact-root point estimates',bundle['estimates'],exact_all(ones))
near('all36 joint covariance versus independent HH finite differences',bundle['covariance'],directcov(F),5e-9)
near('zero-domain HH exact derivatives',F[[2,5,8]],np.zeros((3,36)),1e-13)
check('full-frame df retains zero-domain HH',bundle['df']==6)
check('paired full-block identity',bundle['checks']['identity_pass'])
check('full covariance PSD and symmetric',all(bundle['checks'][part][k] for part in ['joint','target'] for k in ['finite','symmetric','psd']))
# Demonstrate positive covariance makes independent variance addition inappropriate.
jd=bundle['target_ids'].index('D_release_i1_i5');jp=bundle['target_ids'].index('P_release_i1_i5');baseidx=bundle['joint_columns'].index('AS01_S4:i5')
wrong=bundle['covariance'][jd,jd]+bundle['joint_covariance'][baseidx,baseidx]
check('paired variance differs from independent sum',abs(wrong-bundle['covariance'][jp,jp])>1e-4)
# Row order and whole-household tie sensitivity.
order=rng.permutation(n);permuted={k:{**v,'influence':v['influence'][order]} for k,v in models.items()}
near('row permutation covariance',assemble(permuted,targets,hi[order],ss)['covariance'],bundle['covariance'],1e-12)
st=ss.copy();st[[1,4]]=st[[4,1]]
tie=assemble(models,targets,hi,st)
near('tie leaves point estimates identical',tie['estimates'],bundle['estimates'],0)
check('tie can change covariance without changing points',np.max(np.abs(tie['covariance']-bundle['covariance']))>1e-6)
# All gates start passing on exact-root synthetic model; quadrature variants identical for algebra test.
bundles={v:copy.deepcopy(bundle) for v in VARIANTS}
tchecks={t['id']:target_gate(bundles,t['id'])[0] for t in targets}
mc={name:{'synthetic':True} for name in models}
flags,_=release_decisions(mc,tchecks,targets);check('all36 pass positive synthetic gates',all(flags.values()))
fail=copy.deepcopy(tchecks);fail['P_release_i1_i5']['q121_se_stability']=False
flags,_=release_decisions(mc,fail,targets);check('one paired failure holds all4 paired only',sum(flags.values())==32 and all(flags[t['id']] for t in targets if t['kind']=='direct'))
badmc=copy.deepcopy(mc);badmc['AS01_S4']['synthetic']=False
flags,_=release_decisions(badmc,tchecks,targets);check('base failure holds all16paired not20direct',sum(flags.values())==20 and all(flags[t['id']]==(t['kind']=='direct') for t in targets))
badmc=copy.deepcopy(mc);badmc['release_i4']['synthetic']=False
flags,_=release_decisions(badmc,tchecks,targets);check('release model failure holds its9 targets',sum(flags.values())==27)
fail=copy.deepcopy(tchecks);fail['D_release_i2_i2']['q81_positive_resolution']=False
flags,_=release_decisions(mc,fail,targets);check('one direct failure holds its5direct only',sum(flags.values())==31)
# Very correlated IF arrays: primary subtraction retains tiny difference; guard prevents near-zero precision claim.
close=copy.deepcopy(models);base=close['AS01_S4']
for name,v in close.items():
    if name=='AS01_S4':continue
    v['influence'][:,:4]=base['influence']+1e-12*rng.normal(size=(n,4))
    v['correction'][:4]=base['correction']
cb=assemble(close,targets,hi,ss);cg,_=target_gate({v:cb for v in VARIANTS},'P_release_i1_i5')
check('near-zero paired SE withheld',not cg['q81_positive_resolution'])
check('near-zero covariance identity within fixed absolute tolerance',cb['checks']['identity_pass'])
changed=copy.deepcopy(bundles);changed['q121']['standard_errors']*=1.03
ch,_=target_gate(changed,'P_release_i1_i5');check('Q instability detected',not ch['q121_se_stability'])
changed=copy.deepcopy(bundles);changed['q81half']['standard_errors']*=1.03
ch,_=target_gate(changed,'D_release_i3_i3');check('half-step instability detected',not ch['q81half_se_stability'])
changed=copy.deepcopy(bundles);changed['q81']['correction'][:]=1
ch,_=target_gate(changed,'P_release_i1_i5');check('paired Newton failure detected',not ch['q81_newton'])
changed=copy.deepcopy(bundles);changed['q121']['checks']['identity_pass']=False
ch,_=target_gate(changed,'P_release_i1_i5');check('covariance identity failure detected',not ch['q121_identity'])
missing={**models,'release_i4':None};mb=assemble(missing,targets,hi,ss)
check('missing model omits only its9 targets from covariance',len(mb['target_ids'])==27)
ch,_=target_gate({v:mb for v in VARIANTS},'D_release_i4_i4');check('missing model unavailable not zero SE',ch=={'available':False})
check('family retains36 even with unavailable targets',len(targets)==plan['intervals']['family_size']==36)
md={v:dict(finite=True,gradient_max=1e-5,hessian_min_eigenvalue=.01,hessian_asymmetry=1e-9,hessian_max_abs=2,score_identity_error=1e-12,bound_hits=[]) for v in VARIANTS}
check('model valid gates',all(model_gate(md).values()))
for field,value,gate in [('finite',False,'finite'),('gradient_max',.001,'stationary'),('hessian_min_eigenvalue',0,'hessian'),('hessian_asymmetry',.001,'asymmetry'),('score_identity_error',1e-6,'score_identity'),('bound_hits',['x'],'bounds')]:
    dd=copy.deepcopy(md);dd['q121'][field]=value;check('model rejects '+field,not model_gate(dd)['q121_'+gate])
# Actual MNLFA common-reference Jacobian: independent common_reference and central difference all five parameterizations.
maxerr=0.
for released in [None,0,3,1,2]:
    anchors=[j for j in [0,1,2,3] if j!=released];m=Model('slope_adjusted',9,anchors=anchors)
    theta=rng.normal(0,.1,m.p);xx=np.array([[0,-1,0],[1,1,1],[1,0,0],[0,2,1]],float);items=[4,5,6,7]+([] if released is None else [released])
    values,jac=reference(m,theta,xx,items)
    near('MNLFA probability match release '+str(released),values,np.array([m.common_reference(theta,x)[items] for x in xx]),1e-13)
    for j in range(m.p):
        up=theta.copy();down=theta.copy();up[j]+=1e-6;down[j]-=1e-6
        derivative=(reference(m,up,xx,items)[0]-reference(m,down,xx,items)[0])/2e-6
        err=float(np.max(np.abs(derivative-jac[:,:,j])));maxerr=max(maxerr,err)
        check('MNLFA Jacobian '+str(released)+' '+str(j),err<2e-9)
    vals,J=inputs(m,theta,xx,np.array([.5,2.,3.,.9]),items)
    check('MNLFA target dimensions '+str(released),vals.shape==(4,len(items)) and J.shape==(len(items),m.p))
# Regression checks for independent review findings.
paired_ix=[j for j,t in enumerate(targets) if t['kind']=='paired_change']
expected_tol=1e-14+1e-8*np.max(np.abs(bundle['covariance'][np.ix_(paired_ix,paired_ix)]))
near('paired identity tolerance uses paired covariance scale',bundle['checks']['identity_tolerance'],expected_tol,1e-25)
check('paired scale is materially smaller than direct family scale',np.max(np.abs(bundle['covariance'][np.ix_(paired_ix,paired_ix)]))<np.max(np.abs(bundle['covariance']))/10)
tie_changes={v:copy.deepcopy(tie) for v in VARIANTS}
tie_changes['q121']['standard_errors']*=1.1
tie_tc={t['id']:target_gate(tie_changes,t['id'])[0] for t in targets}
tie_flags,_=release_decisions(mc,tie_tc,targets)
primary_flags,_=release_decisions(mc,tchecks,targets)
check('tie gates independently hold failing sensitivity despite passing primary',all(primary_flags.values()) and not any(tie_flags.values()))
from run_joint import worker_failure
name,unavailable,failed=worker_failure('release_i4',FloatingPointError('synthetic failure'),persist=False)
check('worker failure retains variant slots without fake results',all(v is None for v in unavailable.values()) and failed=={'worker_completed':False})
failedmc={**mc,name:failed};mbs={v:assemble({**models,name:unavailable[v]},targets,hi,ss) for v in VARIANTS}
mtc={t['id']:target_gate(mbs,t['id'])[0] for t in targets}
mflags,mreasons=release_decisions(failedmc,mtc,targets)
check('worker failure keeps36rows and releases remaining27',len(mflags)==36 and sum(mflags.values())==27 and all(mreasons[t['id']] for t in targets if t['model']==name))
try:worker_failure('release_i4',AssertionError('synthetic integrity failure'),persist=False)
except AssertionError:check('unexpected integrity error is not model exclusion',True)
else:check('unexpected integrity error is not model exclusion',False)
changed=copy.deepcopy(bundles);changed['q81']['checks']['identity_pass']=False
ctc={t['id']:target_gate(changed,t['id'])[0] for t in targets};cf,_=release_decisions(mc,ctc,targets)
check('paired identity failure holds paired not unaffected direct intervals',sum(cf.values())==20 and all(cf[t['id']] for t in targets if t['kind']=='direct'))
result=dict(status='PASS' if all(c['passed'] for c in checks) else 'FAIL',checks=checks,passed=sum(c['passed'] for c in checks),total=len(checks),seed=2026102502,max_reference_jacobian_error=maxerr,real_data_read=False,optimization_calls=0,participant_arrays_exported=0,completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in ['joint.py','run_joint.py','analysis_plan.json','tests_joint.py']})
(ROOT/'review/synthetic_checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='checks'}));assert result['status']=='PASS',[c for c in checks if not c['passed']]
