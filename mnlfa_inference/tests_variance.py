"""Synthetic-only independent tests for variance.py; no participant inputs.

The optional JSON report contains test summaries and source hashes only.
Numerical derivatives use an independently written household-multiplier
algorithm that recomputes n_g, W_g and N, not the influence formula under test.
"""
from pathlib import Path
import argparse, datetime, hashlib, json, platform, sys
sys.dont_write_bytecode = True
import numpy as np
import scipy
from scipy.stats import t as student_t
from variance import normalized_mean_influence, household_covariance, sandwich_targets, CONDITIONING


def run_tests():
    checks=[]
    def check(name,passed,details=None):
        row={'name':name,'passed':bool(passed)}
        if details is not None: row['details']=details
        checks.append(row)
    def near(name,actual,expected,tolerance=1e-9):
        error=float(np.max(np.abs(np.asarray(actual)-np.asarray(expected))))
        check(name,np.isfinite(error) and error<=tolerance,{'max_absolute_difference':error,'tolerance':tolerance})
    def rejects(name,fn):
        try:
            fn()
        except (ValueError,FloatingPointError):
            check(name,True)
        else:
            check(name,False)

    rng=np.random.default_rng(20261025)
    group=np.array([0,1,0,0,1,1,0,1,1,0,1,1])
    weights=np.array([.4,1.4,2.3,.7,2.4,3.,.6,1.7,.8,2.,1.9,.3])
    domain_households=np.array([0,0,1,1,1,3,3,4,4,6,6,7])
    household_strata=np.repeat(['A','B','C'],3)
    # Full synthetic frame explicitly includes domain-excluded members in
    # mixed households0/3 and completely out-of-domain households2/5/8.
    full_household=np.r_[domain_households,[0,3,2,5,8,8]]
    domain=np.r_[np.ones(12,dtype=bool),np.zeros(6,dtype=bool)]
    H=len(household_strata);n=len(group);p=3;k=2
    scores=rng.normal(size=(n,p))+np.array([1.,-.3,.6])
    d=rng.normal(size=(n,k))
    hessian=np.array([[1.4,.3,-.1],[.3,1.1,.2],[-.1,.2,1.8]])
    jacobian=np.array([[.4,-.2,.15],[-.1,.35,.2]])
    fixed_jacobian=np.array([[.2,.1,-.3],[.5,0.,0.]])
    fixed_estimates=np.array([.1,-.08])
    one=np.ones(H)

    def independent_mean(value,multiplier):
        # This implementation describes the actual reweighted estimand.
        # Domain membership is applied to the complete synthetic frame first.
        case_multiplier=multiplier[full_household][domain]
        total=case_multiplier.sum()
        output=np.zeros(value.shape[1])
        for label in sorted(set(group.tolist())):
            ix=group==label
            ng=case_multiplier[ix].sum()
            wg=(case_multiplier[ix]*weights[ix]).sum()
            avg=((case_multiplier[ix]*weights[ix])[:,None]*value[ix]).sum(axis=0)/wg
            output+=ng/total*avg
        return output

    def fd_households(function,step=2e-5):
        answer=[]
        for hh in range(H):
            hi=one.copy();lo=one.copy();hi[hh]+=step;lo[hh]-=step
            answer.append((function(hi)-function(lo))/(2*step))
        return np.asarray(answer)

    def independent_covariance(household_if):
        covariance=np.zeros((household_if.shape[1],household_if.shape[1]))
        for label in ('A','B','C'):
            vectors=household_if[household_strata==label]
            average=sum(vectors)/len(vectors)
            for vector in vectors:
                centered=vector-average
                covariance+=len(vectors)/(len(vectors)-1)*np.outer(centered,centered)
        return covariance

    score_result=normalized_mean_influence(scores,group,weights)
    reference_result=normalized_mean_influence(d,group,weights)
    near('mean score equals independent dynamic normalization',score_result['mean'],independent_mean(scores,one))
    near('reference mean equals independent dynamic normalization',reference_result['mean'],independent_mean(d,one))
    check('nonzero mean score exercised',np.max(abs(score_result['mean']))>.1)
    near('score influence sums to zero',score_result['row_influence'].sum(0),np.zeros(p))
    near('reference influence sums to zero',reference_result['row_influence'].sum(0),np.zeros(k))
    score_hh=household_covariance(score_result['row_influence'],domain_households,household_strata)
    reference_hh=household_covariance(reference_result['row_influence'],domain_households,household_strata)
    fd_score=fd_households(lambda m:independent_mean(scores,m))
    fd_reference=fd_households(lambda m:independent_mean(d,m))
    near('full-frame HH score derivative with count and W renormalization',score_hh['household_influence'],fd_score,2e-9)
    near('full-frame HH empirical-reference derivative',reference_hh['household_influence'],fd_reference,2e-9)
    near('same-household cross-hearing rows aggregated',score_hh['household_influence'][0],score_result['row_influence'][0]+score_result['row_influence'][1])
    near('mixed household derivative ignores its domain-excluded member',score_hh['household_influence'][3],score_result['row_influence'][5]+score_result['row_influence'][6])
    near('completely excluded households retain zero IF',score_hh['household_influence'][[2,5,8]],np.zeros((3,p)))
    check('complete household frame and df retained',score_hh['households']==9 and score_hh['strata']==3 and score_hh['df']==6 and score_hh['zero_domain_households']==3)
    near('stratum covariance independent finite-difference calculation',score_hh['covariance'],independent_covariance(fd_score),2e-9)
    check('finite-cluster correction is not FPC',score_hh['finite_cluster_correction'] and not score_hh['sampling_fpc'])
    check('conditioning string explicit',score_result['conditioning']==CONDITIONING)

    # Separately perturb each CASE, including a point with a nonzero U, so a
    # missing -U term or use of fixed n_g fails this derivative comparison.
    def independent_case_mean(value,m):
        total=m.sum();answer=np.zeros(value.shape[1])
        for g in (0,1):
            ix=group==g
            answer+=(m[ix].sum()/total)*np.average(value[ix],axis=0,weights=m[ix]*weights[ix])
        return answer
    for j in range(n):
        hi=np.ones(n);lo=np.ones(n);hi[j]+=2e-5;lo[j]-=2e-5
        fd=(independent_case_mean(scores,hi)-independent_case_mean(scores,lo))/(4e-5)
        near('independent score case derivative '+str(j),score_result['row_influence'][j],fd,2e-9)

    result=sandwich_targets(scores,group,weights,hessian,d,jacobian,domain_households,household_strata,fixed_estimates,fixed_jacobian)
    near('negative-score sandwich sign',result['theta_row_influence']@hessian.T,-score_result['row_influence'])
    near('empirical total IF includes both terms',result['target_row_influence'][:,:k],reference_result['row_influence']+result['theta_row_influence']@jacobian.T)
    near('fixed profiles have zero direct empirical term',result['direct_row_influence'][:,k:],np.zeros((n,2)))
    near('fixed profiles parameter influence only',result['target_row_influence'][:,k:],result['theta_row_influence']@fixed_jacobian.T)
    near('full covariance from direct and parameter cross terms',result['covariance'],result['direct_covariance']+result['parameter_covariance']+result['direct_parameter_cross_covariance']+result['direct_parameter_cross_covariance'].T)
    check('nonzero cross term is exercised',np.max(abs(result['direct_parameter_cross_covariance']))>1e-5)
    check('independent variance addition is demonstrably wrong',np.max(abs(result['covariance']-result['direct_covariance']-result['parameter_covariance']))>1e-5)
    near('theta sandwich is inverse Hessian score covariance inverse Hessian',result['theta_covariance'],np.linalg.solve(hessian,np.linalg.solve(hessian,score_hh['covariance']).T).T,2e-10)
    near('t intervals use design df not normal critical value',result['critical_t'],student_t.ppf(.975,6),1e-14)
    near('t lower interval',result['intervals'][:,0],result['estimates']-student_t.ppf(.975,6)*result['standard_errors'])
    near('t upper interval',result['intervals'][:,1],result['estimates']+student_t.ppf(.975,6)*result['standard_errors'])
    near('covariance symmetric',result['covariance'],result['covariance'].T)
    eig=np.linalg.eigvalsh(result['covariance'])
    check('covariance positive semidefinite',eig.min()>=-1e-12,{'minimum_eigenvalue':float(eig.min())})

    # An analytically solvable synthetic estimating equation supplies an
    # independent TOTAL derivative including target-parameter covariance.
    b=rng.normal(size=(n,p))
    theta0=-np.linalg.solve(hessian,independent_mean(b,one))
    base_scores=b+theta0@hessian.T
    base_values=rng.normal(size=(n,k))
    row_jacobian=rng.normal(0,.25,size=(n,k,p))
    values0=base_values+np.einsum('nkp,p->nk',row_jacobian,theta0)
    mean_jacobian=independent_mean(row_jacobian.reshape(n,k*p),one).reshape(k,p)
    fixed_base=np.array([.2,-.1])
    fixed0=fixed_base+fixed_jacobian@theta0
    implicit=sandwich_targets(base_scores,group,weights,hessian,values0,mean_jacobian,domain_households,household_strata,fixed0,fixed_jacobian)
    def exact_theta(multiplier):
        return -np.linalg.solve(hessian,independent_mean(b,multiplier))
    def exact_targets(multiplier):
        theta=exact_theta(multiplier)
        values=base_values+np.einsum('nkp,p->nk',row_jacobian,theta)
        return np.r_[independent_mean(values,multiplier),fixed_base+fixed_jacobian@theta]
    near('synthetic exact estimating root is centered',implicit['estimating_equation'],np.zeros(p))
    near('implicit theta HH derivative versus exact analytical perturbed root',implicit['theta_household_influence'],fd_households(exact_theta),2e-9)
    fd_total=fd_households(exact_targets)
    near('total target HH derivative including dynamic empirical distribution',implicit['target_household_influence'],fd_total,3e-9)
    near('joint target covariance versus fully independently perturbed estimator',implicit['covariance'],independent_covariance(fd_total),3e-9)

    # Original weights may be rescaled globally or independently within group.
    for name,scaled in (('global',weights*9.7),('group-specific',weights*np.where(group==0,13.,.27))):
        changed=sandwich_targets(scores,group,scaled,hessian,d,jacobian,domain_households,household_strata,fixed_estimates,fixed_jacobian)
        near(name+' raw-weight scale invariance estimates',changed['estimates'],result['estimates'])
        near(name+' raw-weight scale invariance score IF',changed['theta_row_influence'],result['theta_row_influence'])
        near(name+' raw-weight scale invariance joint covariance',changed['covariance'],result['covariance'])

    # Constant response contrasts represent hearing_only/latent_adjusted
    # targets: latent impacts must not create empirical-reference variation.
    constant=np.tile([.021,-.014],(n,1))
    const=normalized_mean_influence(constant,group,weights)
    near('constant reference contrasts have zero empirical IF',const['row_influence'],np.zeros((n,k)),2e-16)
    # J for a target can be zero on a latent parameter, but A^-1 allows that
    # parameter score to contribute through estimated-parameter covariance.
    latent_only_scores=np.column_stack((np.zeros(n),scores[:,1],np.zeros(n)))
    one_profile=sandwich_targets(latent_only_scores,group,weights,hessian,None,None,domain_households,household_strata,[.1],[[1.,0.,0.]])
    check('zero direct latent target derivative does not erase covariance path',one_profile['standard_errors'][0]>1e-4)

    # Excluded-only households affect stratum centering and H/(H-1); dropping
    # them is a different variance estimator even though their IF is zero.
    occupied=np.unique(domain_households)
    reindex=np.full(H,-1);reindex[occupied]=np.arange(len(occupied))
    dropped=household_covariance(result['target_row_influence'],reindex[domain_households],household_strata[occupied])
    check('dropping zero-domain households changes variance',np.max(abs(dropped['covariance']-result['covariance']))>1e-6)
    check('dropping zero-domain households changes df',dropped['df']!=result['df'])
    # A row permutation changes neither aggregation nor result.
    order=rng.permutation(n)
    perm=sandwich_targets(scores[order],group[order],weights[order],hessian,d[order],jacobian,domain_households[order],household_strata,fixed_estimates,fixed_jacobian)
    near('row permutation invariant covariance',perm['covariance'],result['covariance'])
    near('row permutation invariant estimates',perm['estimates'],result['estimates'])
    rejects('negative original weight rejected',lambda:normalized_mean_influence(scores,group,-weights))
    rejects('zero original weight rejected',lambda:normalized_mean_influence(scores,group,np.r_[0.,weights[1:]]))
    rejects('NaN scores rejected',lambda:normalized_mean_influence(np.full_like(scores,np.nan),group,weights))
    rejects('out-of-frame HH index rejected',lambda:household_covariance(scores,np.full(n,H),household_strata))
    rejects('fractional HH index rejected',lambda:household_covariance(scores,domain_households+.5,household_strata))
    rejects('lonely stratum rejected without automatic adjustment',lambda:household_covariance(scores,domain_households,np.r_[['X'],household_strata[1:]]))
    rejects('nonpositive Hessian rejected without pseudoinverse',lambda:sandwich_targets(scores,group,weights,-hessian,d,jacobian,domain_households,household_strata))
    bad=hessian.copy();bad[0,1]+=.2
    rejects('asymmetric Hessian rejected',lambda:sandwich_targets(scores,group,weights,bad,d,jacobian,domain_households,household_strata))
    rejects('empirical Jacobian dimensions rejected',lambda:sandwich_targets(scores,group,weights,hessian,d,np.ones((4,p)),domain_households,household_strata))
    rejects('invalid interval alpha rejected',lambda:sandwich_targets(scores,group,weights,hessian,d,jacobian,domain_households,household_strata,alpha=1.))
    rejects('missing fixed estimates rejected',lambda:sandwich_targets(scores,group,weights,hessian,d,jacobian,domain_households,household_strata,fixed_jacobian=fixed_jacobian))
    report={'status':'PASS' if all(c['passed'] for c in checks) else 'FAIL',
            'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'checks_passed':sum(c['passed'] for c in checks),'checks_total':len(checks),'checks':checks,
            'runtime':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__},
            'seed':20261025,'synthetic_domain_rows':n,'synthetic_full_frame_rows':len(domain),
            'synthetic_full_households':H,'synthetic_strata':3,
            'domain_excluded_households':3,'mixed_domain_households':2,'cross_hearing_household_present':True,
            'conditioning':CONDITIONING,'real_data_read':False,'real_models_fitted':0,'optimization_calls':0,
            'participant_arrays_exported':False,
            'limitations':['These algebraic/synthetic checks do not validate the household/GOR adapter as the original sampling design.',
                           'A valid real analysis still requires scores and observed Hessian evaluated at the same theta/Q, model numerical checks, and independent result QA.',
                           'Intervals are pointwise t approximations and do not control multiplicity or uncertainty in anchor/model selection.']}
    here=Path(__file__).resolve().parent
    report['source_hashes']={name:hashlib.sha256((here/name).read_bytes()).hexdigest() for name in ('variance.py','tests_variance.py')}
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'tests_variance_results.json')
    args=parser.parse_args()
    result=run_tests()
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'status':result['status'],'checks_passed':result['checks_passed'],'checks_total':result['checks_total'],
                      'failed':[c for c in result['checks'] if not c['passed']],'report':str(args.output)},indent=2))
    raise SystemExit(0 if result['status']=='PASS' else 1)
