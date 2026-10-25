"""Joint model uncertainty: pure RAM algebra, no file/network/fit access."""
import numpy as np
from scipy.special import ndtr
from variance import normalized_mean_influence, household_covariance
from engine import X_NAMES

VARIANTS=('q81','q121','q81half')

def reference(m,theta,xx,items):
    xx=np.atleast_2d(np.asarray(xx,float));items=np.asarray(items,int)
    c=m.to_components(theta);b=c['b'][None,:]+xx@c['delta'].T
    a2=np.exp(2*(c['loga'][None,:]+xx@c['omega'].T));den=np.sqrt(1+a2);z=b/den
    pdf=np.exp(-z*z/2)/np.sqrt(2*np.pi);db=pdf/den;dl=-pdf*b*a2/den**3
    jac=np.zeros((len(xx),len(items),m.p));lookup={n:k for k,n in enumerate(m.names)}
    for l,j in enumerate(items):
        jac[:,l,lookup[f'b_i{j+1}']]=db[:,j];jac[:,l,lookup[f'loga_i{j+1}']]=dl[:,j]
        for k,name in enumerate(X_NAMES):
            for prefix,derivative in [('intercept',db),('logslope',dl)]:
                n=f'{prefix}_i{j+1}_{name}'
                if n in lookup:jac[:,l,lookup[n]]=derivative[:,j]*xx[:,k]
    return ndtr(z)[:,items],jac

def inputs(m,theta,x,weights,items):
    profiles,inverse=np.unique(x[:,1:],axis=0,return_inverse=True)
    v0,j0=reference(m,theta,np.column_stack((np.zeros(len(profiles)),profiles)),items)
    v1,j1=reference(m,theta,np.column_stack((np.ones(len(profiles)),profiles)),items)
    values=(v1-v0)[inverse];jr=(j1-j0)[inverse]
    J=normalized_mean_influence(jr.reshape(len(x),-1),x[:,0],weights)['mean'].reshape(len(items),m.p)
    return values,J

def linearize(scores,group,weights,A,values,J):
    u=normalized_mean_influence(scores,group,weights)
    v=normalized_mean_influence(values,group,weights)
    theta_if=-np.linalg.solve(A,u['row_influence'].T).T
    influence=v['row_influence']+theta_if@J.T
    correction=J@(-np.linalg.solve(A,u['mean']))
    return dict(estimates=v['mean'],influence=influence,correction=correction)

def cov_checks(V):
    V=np.asarray(V,float)
    finite=bool(np.isfinite(V).all());scale=float(np.max(np.abs(V))) if finite else 0.
    asym=float(np.max(np.abs(V-V.T))) if finite else float('inf')
    eig=np.linalg.eigvalsh(V) if finite else np.array([-1.,1.])
    return dict(finite=finite,symmetric=bool(asym<=1e-14+1e-8*scale),
                psd=bool(eig[0]>=-1e-10*max(float(eig[-1]),1e-16)),
                asymmetry=asym,minimum_eigenvalue=float(eig[0]),maximum_eigenvalue=float(eig[-1]))

def assemble(models,targets,hi,strata):
    """models order is base then four releases; missing model keeps affected targets unavailable."""
    active={k:v for k,v in models.items() if v is not None}
    if not active:return None
    columns=[];offset={};blocks=[]
    for key,v in active.items():
        for j,item in enumerate(v['items']):
            offset[(key,item)]=len(columns);columns.append(key+':'+item)
        blocks.append(v['influence'])
    full_if=np.column_stack(blocks);joint=household_covariance(full_if,hi,strata)
    T=[];target_if=[];ids=[];est=[];corr=[]
    for t in targets:
        key=t['model'];item=t['item'];paired=t['kind']=='paired_change'
        if key not in active or (paired and 'AS01_S4' not in active):continue
        model=active[key];j=model['items'].index(item);row=np.zeros(len(columns));row[offset[key,item]]=1
        influence=model['influence'][:,j].copy();estimate=model['estimates'][j];correction=model['correction'][j]
        if paired:
            base=active['AS01_S4'];j0=base['items'].index(item);row[offset['AS01_S4',item]]=-1
            influence-=base['influence'][:,j0];estimate-=base['estimates'][j0];correction-=base['correction'][j0]
        T.append(row);target_if.append(influence);ids.append(t['id']);est.append(estimate);corr.append(correction)
    if not ids:return None
    V=household_covariance(np.column_stack(target_if),hi,strata)['covariance'];T=np.array(T)
    transformed=T@joint['covariance']@T.T
    full_error=float(np.max(np.abs(V-transformed)))
    paired=[j for j,ident in enumerate(ids) if next(t for t in targets if t['id']==ident)['kind']=='paired_change']
    if paired:
        paired_cov=V[np.ix_(paired,paired)];paired_transformed=transformed[np.ix_(paired,paired)]
        error=float(np.max(np.abs(paired_cov-paired_transformed)))
        tol=1e-14+1e-8*float(np.max(np.abs(paired_cov)))
    else:error=0.;tol=1e-14
    full_tol=1e-14+1e-8*float(np.max(np.abs(V)))
    diag=np.diag(V);se=np.sqrt(np.maximum(diag,0)) # nonpositive diagonals are withheld, never repaired/released
    return dict(target_ids=ids,estimates=np.array(est),correction=np.array(corr),covariance=V,
                standard_errors=se,positive_variances=bool(np.all(diag>0)),joint_columns=columns,
                joint_covariance=joint['covariance'],transformation=T,df=joint['df'],
                checks={'joint':cov_checks(joint['covariance']),'target':cov_checks(V),
                        'identity_pass':bool(error<=tol and full_error<=full_tol),'identity_error':error,'identity_tolerance':tol,
                        'identity_scope':'paired-difference covariance; full-family check also required',
                        'full_identity_error':full_error,'full_identity_tolerance':full_tol})

def model_gate(details):
    checks={}
    for key in VARIANTS:
        d=details[key]
        checks[key+'_finite']=bool(d['finite'])
        checks[key+'_stationary']=d['gradient_max']<=2e-4
        checks[key+'_hessian']=d['hessian_min_eigenvalue']>1e-7
        checks[key+'_asymmetry']=d['hessian_asymmetry']<=1e-7*max(1.,d['hessian_max_abs'])
        checks[key+'_score_identity']=d['score_identity_error']<=1e-10
        checks[key+'_bounds']=not d['bound_hits']
    return checks

def target_gate(bundles,target_id):
    checks={};d={}
    for key in VARIANTS:
        b=bundles[key]
        if b is None or target_id not in b['target_ids']:return {'available':False},{}
        j=b['target_ids'].index(target_id);se=float(b['standard_errors'][j]);cor=float(b['correction'][j])
        d[key]=dict(se=se,newton_correction=cor)
        checks[key+'_positive_resolution']=bool(np.isfinite(se) and se>1e-8 and b['covariance'][j,j]>0)
        checks[key+'_newton']=bool(np.isfinite(cor) and abs(cor)<=max(1e-5,.05*se))
        checks[key+'_covariance']=all(b['checks']['joint'][k] and b['checks']['target'][k] for k in ('finite','symmetric','psd'))
        checks[key+'_identity']=b['checks']['identity_pass'] if target_id.startswith('P_') else True
    for comparator in ['q121','q81half']:
        a=d['q81']['se'];c=d[comparator]['se'];absolute=abs(a-c);den=max(c,1e-8);relative=absolute/den
        d['q81_vs_'+comparator]=dict(absolute_se_change=absolute,denominator=den,relative_se_change=relative)
        checks[comparator+'_se_stability']=bool(np.isfinite(relative) and relative<=.02)
    return checks,d

def release_decisions(model_checks,target_checks,targets):
    released={};reasons={}
    for t in targets:
        model=t['model'];kind=t['kind'];related=[a for a in targets if a['model']==model and a['kind']==kind]
        why=[model+':'+k for k,v in model_checks[model].items() if not v]
        if kind=='paired_change':why += ['AS01_S4:'+k for k,v in model_checks['AS01_S4'].items() if not v]
        for r in related:why += [r['id']+':'+k for k,v in target_checks[r['id']].items() if not v]
        released[t['id']]=not why;reasons[t['id']]=why
    return released,reasons
