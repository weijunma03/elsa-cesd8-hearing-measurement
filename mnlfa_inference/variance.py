"""Local-RAM linearization for group-normalized MNLFA estimating equations.

This module has no file, data, optimization or network access. Arrays returned
as *_influence are participant/household level working arrays: callers must
keep them in RAM and export only aggregate covariance/diagnostic summaries.

The resampling estimand recomputes domain group counts, group weight sums and
the domain total under each household multiplier. It does NOT condition on
the previously normalized weights. Household/GOR is a declared design adapter,
not a claim that these are the released original sampling PSUs/strata.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import t as student_t

CONDITIONING = 'dynamically_recomputed_group_counts_and_weight_sums'


def _matrix(values, name, n=None):
    result = np.asarray(values, dtype=float)
    if result.ndim == 1:
        result = result[:, None]
    if result.ndim != 2 or (n is not None and result.shape[0] != n):
        raise ValueError(name+' must have the expected two-dimensional shape')
    if not np.isfinite(result).all():
        raise ValueError(name+' contains nonfinite values')
    return result


def _labels(values, name, n):
    result = np.asarray(values)
    if result.ndim != 1 or len(result) != n:
        raise ValueError(name+' must be a vector of the expected length')
    if np.issubdtype(result.dtype, np.number):
        if not np.isfinite(result).all():
            raise ValueError(name+' contains nonfinite labels')
    elif result.dtype.kind in 'US':
        if np.any(result == ''):
            raise ValueError(name+' contains empty labels')
    elif result.dtype.kind == 'O':
        if any(v is None or (isinstance(v, (float, np.floating)) and not np.isfinite(v)) for v in result):
            raise ValueError(name+' contains missing labels')
    try:
        unique, inverse = np.unique(result, return_inverse=True)
    except TypeError as exc:
        raise ValueError(name+' must contain mutually comparable scalar labels') from exc
    return result, unique, inverse


def normalized_mean_influence(values, group, weights):
    """Mean and case-multiplier derivative with n_g/W_g recomputed.

    values is n x k, e.g. unweighted individual negative-loglikelihood scores
    or individual common-reference contrasts. weights are the original
    positive domain weights, never the previously normalized weights.

    Returns mean[k], row_influence[n,k], group_counts[G], group_weight_sums[G],
    group_means[G,k] and normalization[G]. For case i in group g,
    IF_i = (bar_v_g - mean + c_g*w_i*(v_i-bar_v_g))/N.
    Group labels and row arrays are RAM-only returned diagnostics.
    """
    values = _matrix(values, 'values')
    n, k = values.shape
    if n < 1 or k < 1:
        raise ValueError('At least one domain row and one value column required')
    _, labels, inverse = _labels(group, 'group', n)
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (n,) or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError('Original weights must be finite strictly positive n-vector')
    count = np.bincount(inverse, minlength=len(labels)).astype(float)
    weight_sum = np.bincount(inverse, weights=weights, minlength=len(labels))
    if not np.isfinite(weight_sum).all() or np.any(weight_sum <= 0):
        raise ValueError('Invalid group weight sums')
    weighted_sum = np.zeros((len(labels), k))
    np.add.at(weighted_sum, inverse, values*weights[:, None])
    means = weighted_sum/weight_sum[:, None]
    c = count/weight_sum
    mean = (count[:, None]*means).sum(axis=0)/n
    centered_group = values-means[inverse]
    row_if = (means[inverse]-mean[None, :]
              +(c[inverse]*weights)[:, None]*centered_group)/n
    if not np.isfinite(mean).all() or not np.isfinite(row_if).all():
        raise FloatingPointError('Nonfinite normalized mean or influence')
    return {'mean': mean, 'row_influence': row_if, 'group_labels': labels,
            'group_counts': count.astype(int), 'group_weight_sums': weight_sum,
            'group_means': means, 'normalization': c,
            'conditioning': CONDITIONING, 'n_domain': n,
            'maximum_absolute_influence_sum': float(np.max(np.abs(row_if.sum(axis=0))))}


def household_covariance(row_influence, household_index, household_stratum):
    """Aggregate all domain rows within household, then stratified WR covariance.

    household_index is an integer vector indexing rows of the COMPLETE
    household skeleton household_stratum, including households with no domain
    members. Cross-group members must carry the same household index. No
    participant or household identifier is needed. All skeleton households are
    retained; their initialized totals are zero when no domain rows map to them.

    V = sum_h H_h/(H_h-1) sum_c (T_hc-mean_h)(T_hc-mean_h)'.
    H_h/(H_h-1) is a finite-cluster correction, not a sampling FPC.
    Degrees of freedom = complete-frame households minus nonempty strata.
    """
    influence = _matrix(row_influence, 'row_influence')
    n, k = influence.shape
    strata = np.asarray(household_stratum)
    if strata.ndim != 1 or len(strata) < 2:
        raise ValueError('Complete household_stratum skeleton required')
    H = len(strata)
    _, labels, si = _labels(strata, 'household_stratum', H)
    index = np.asarray(household_index)
    if index.shape != (n,) or not np.issubdtype(index.dtype, np.number) or not np.isfinite(index).all():
        raise ValueError('household_index must be a finite numeric domain n-vector')
    if np.any(index != np.floor(index)) or np.any(index < 0) or np.any(index >= H):
        raise ValueError('household_index must be integer indices inside complete skeleton')
    index = index.astype(np.int64, copy=False)
    counts = np.bincount(si, minlength=len(labels))
    if np.any(counts < 2):
        raise ValueError('Every declared stratum requires at least two complete-frame households; no automatic lonely-PSU adjustment')
    hh_if = np.zeros((H, k))
    np.add.at(hh_if, index, influence)
    covariance = np.zeros((k, k))
    stratum_means = np.zeros((len(labels), k))
    for h in range(len(labels)):
        rows = si == h
        part = hh_if[rows]
        stratum_means[h] = part.mean(axis=0)
        centered = part-stratum_means[h]
        covariance += counts[h]/(counts[h]-1.)*(centered.T@centered)
    covariance = (covariance+covariance.T)/2.
    if not np.isfinite(covariance).all():
        raise FloatingPointError('Nonfinite household covariance')
    occupied = np.bincount(index, minlength=H) > 0
    return {'covariance': covariance, 'household_influence': hh_if,
            'stratum_mean_influence': stratum_means, 'stratum_labels': labels,
            'households_by_stratum': counts, 'households': H,
            'strata': len(labels), 'df': H-len(labels),
            'domain_households': int(occupied.sum()),
            'zero_domain_households': int((~occupied).sum()),
            'finite_cluster_correction': True, 'sampling_fpc': False}


def sandwich_targets(scores, group, weights, hessian,
                     empirical_values, empirical_jacobian,
                     household_index, household_stratum,
                     fixed_estimates=None, fixed_jacobian=None, alpha=.05):
    """Observed-Hessian sandwich plus joint empirical-reference uncertainty.

    scores[n,p] are UNWEIGHTED individual negative-loglik scores evaluated at
    the selected parameter vector and the same quadrature order as hessian A.
    A[p,p] is the observed Hessian of the mean group-normalized criterion.
    empirical_values[n,k] contains d_i(theta); empirical_jacobian[k,p] is the
    derivative of its normalized mean at fixed empirical rows. The k empirical
    targets are followed by any fixed profile targets fixed_estimates[l] with
    fixed_jacobian[l,p]. Profiles have no direct empirical-distribution term.

    thetaIF = -A^{-1}u; empirical totalIF = v + J thetaIF. Covariances of these
    two terms are retained jointly, including their cross terms. Pointwise
    two-sided Student t intervals use full-household-frame df; no clipping,
    multiplicity correction or p-value calculation is performed here.
    """
    scores = _matrix(scores, 'scores')
    n, p = scores.shape
    if p == 0:
        raise ValueError('At least one score column required')
    A = np.asarray(hessian, dtype=float)
    if A.shape != (p,p) or not np.isfinite(A).all():
        raise ValueError('hessian must be finite p x p')
    asymmetry = float(np.max(np.abs(A-A.T)))
    if asymmetry > 1e-7*max(1., float(np.max(np.abs(A)))):
        raise ValueError('hessian is materially asymmetric')
    A = (A+A.T)/2.
    eigen = np.linalg.eigvalsh(A)
    if eigen[0] <= 0:
        raise ValueError('Observed Hessian must be positive definite; no pseudoinverse or ridge fallback')
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError('alpha must be strictly between 0 and 1')
    score_result = normalized_mean_influence(scores, group, weights)
    theta_if = -np.linalg.solve(A, score_result['row_influence'].T).T
    if empirical_values is None:
        if empirical_jacobian is not None:
            raise ValueError('Empirical values and Jacobian must both be supplied')
        empirical_estimates = np.empty(0)
        direct_if = np.empty((n,0))
        J = np.empty((0,p))
        empirical_result = None
    else:
        empirical_result = normalized_mean_influence(_matrix(empirical_values,'empirical_values',n), group, weights)
        empirical_estimates = empirical_result['mean']
        direct_if = empirical_result['row_influence']
        J = np.asarray(empirical_jacobian,dtype=float)
        if J.ndim == 1 and len(empirical_estimates)==1:
            J = J[None,:]
        if J.shape != (len(empirical_estimates),p) or not np.isfinite(J).all():
            raise ValueError('empirical_jacobian must be finite k x p')
    k_empirical = len(empirical_estimates)
    if fixed_estimates is None:
        if fixed_jacobian is not None:
            raise ValueError('Fixed estimates and Jacobian must both be supplied')
        fixed = np.empty(0)
    else:
        fixed = np.atleast_1d(np.asarray(fixed_estimates,dtype=float))
        F = np.asarray(fixed_jacobian,dtype=float)
        if F.ndim == 1 and len(fixed)==1:
            F = F[None,:]
        if fixed.ndim != 1 or not np.isfinite(fixed).all() or F.shape != (len(fixed),p) or not np.isfinite(F).all():
            raise ValueError('Fixed estimates and Jacobian shapes must be l and l x p')
        J = np.vstack((J,F))
        direct_if = np.hstack((direct_if,np.zeros((n,len(fixed)))))
    estimates = np.r_[empirical_estimates,fixed]
    if len(estimates)==0:
        raise ValueError('At least one empirical or fixed target required')
    parameter_if = theta_if@J.T
    target_if = direct_if+parameter_if
    basis = household_covariance(np.hstack((theta_if,direct_if,parameter_if)), household_index, household_stratum)
    total = household_covariance(target_if,household_index,household_stratum)
    cov = total['covariance']
    k = len(estimates)
    direct_cov = basis['covariance'][p:p+k,p:p+k]
    parameter_cov = basis['covariance'][p+k:p+2*k,p+k:p+2*k]
    cross_cov = basis['covariance'][p:p+k,p+k:p+2*k]
    covariance_decomposition = direct_cov+parameter_cov+cross_cov+cross_cov.T
    se = np.sqrt(np.diag(cov))
    critical = float(student_t.ppf(1-alpha/2,total['df']))
    interval = np.column_stack((estimates-critical*se,estimates+critical*se))
    return {'estimates':estimates,'covariance':cov,'standard_errors':se,
            'intervals':interval,'alpha':float(alpha),'critical_t':critical,'df':total['df'],
            'empirical_target_count':k_empirical,'fixed_target_count':len(fixed),
            'theta_covariance':basis['covariance'][:p,:p],
            'direct_covariance':direct_cov,'parameter_covariance':parameter_cov,
            'direct_parameter_cross_covariance':cross_cov,
            'maximum_covariance_decomposition_error':float(np.max(np.abs(cov-covariance_decomposition))),
            'estimating_equation':score_result['mean'],
            'theta_row_influence':theta_if,'direct_row_influence':direct_if,
            'parameter_row_influence':parameter_if,'target_row_influence':target_if,
            'theta_household_influence':basis['household_influence'][:,:p],
            'target_household_influence':total['household_influence'],
            'score_normalization':score_result,'empirical_normalization':empirical_result,
            'household_design':{key:total[key] for key in ('households','strata','households_by_stratum','domain_households','zero_domain_households','finite_cluster_correction','sampling_fpc')},
            'hessian_min_eigenvalue':float(eigen[0]),'hessian_max_eigenvalue':float(eigen[-1]),
            'hessian_asymmetry_max':asymmetry,'conditioning':CONDITIONING,
            'interval_type':'pointwise_two_sided_t_unclipped',
            'participant_arrays_are_ram_only':True}
