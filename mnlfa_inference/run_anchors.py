"""Frozen one-at-a-time AS01 anchor release; aggregate outputs only.

No work occurs on import. Root must freeze and explicitly run this script.
Scientific models, sample, weight normalization, gates and finite refinements
are specified in analysis_plan.json. No survey intervals are computed here.
"""
from pathlib import Path
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import sys
import json
import hashlib
import importlib.util
import datetime
import time
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize
from scipy.integrate import quad_vec
from scipy.special import ndtr
from engine import Model, ANCHORS, NONANCHORS, group_rows

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent.parent
OUT = ROOT / 'results_anchors'
RUN_STARTED = False
START = None
ATTEMPTS = []


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def dump(path, obj):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def progress(status, **kw):
    item = dict(status=status, updated_utc=stamp(), elapsed_seconds=time.time()-START, **kw)
    dump(OUT/'progress.json', item)
    print(json.dumps(item, ensure_ascii=False, allow_nan=False), flush=True)


def guard(event, args):
    if event.startswith('socket.'):
        raise RuntimeError('Network disabled for local anchor sensitivity analysis')


def validate_freeze(root=ROOT):
    manifest = read_json(root/'freeze.json')
    if not isinstance(manifest.get('files'), dict) or not manifest['files']:
        raise RuntimeError('freeze.files must be a nonempty relative-path SHA256 mapping')
    if not isinstance(manifest.get('protected'), dict) or not manifest['protected']:
        raise RuntimeError('freeze.protected must be a nonempty absolute-path SHA256 mapping')
    resolved_root = root.resolve()
    for name, expected in manifest['files'].items():
        path = (root/name).resolve()
        if Path(name).is_absolute() or not path.is_relative_to(resolved_root):
            raise RuntimeError('Frozen relative file leaves new stage: ' + name)
        if sha(path) != expected:
            raise RuntimeError('Frozen file mismatch: ' + name)
    for name, expected in manifest['protected'].items():
        path = Path(name)
        if not path.is_absolute():
            raise RuntimeError('Protected path must be absolute: ' + name)
        if sha(path) != expected:
            raise RuntimeError('Protected source mismatch: ' + name)
    return manifest


def validate_anchor_plan(plan):
    cfg = plan['anchors']
    expected = [('release_i1', 0), ('release_i4', 3), ('release_i2', 1), ('release_i3', 2)]
    if [(r['id'], r['released']) for r in cfg['runs']] != expected:
        raise ValueError('Exactly the four prespecified single-anchor releases are required')
    for run in cfg['runs']:
        if set(run['anchors']) != set(int(j) for j in ANCHORS if j != run['released']) or len(run['anchors']) != 3:
            raise ValueError('Unexpected anchor set for ' + run['id'])
    if cfg['stage'] != 'slope_adjusted' or cfg['parameters_each'] != 60:
        raise ValueError('Anchor sensitivity requires the 60-parameter S4 model')
    if cfg['quadrature_pairs'] != [[61, 81], [81, 121]]:
        raise ValueError('Only frozen 61/81 and 81/121 refinement levels are allowed')
    return cfg


def projected(g, t, bounds):
    g = np.asarray(g).copy()
    for j, (lo, hi) in enumerate(bounds):
        if (t[j] <= lo+1e-8 and g[j] > 0) or (t[j] >= hi-1e-8 and g[j] < 0):
            g[j] = 0.
    return float(np.max(np.abs(g)))


def perturb(t, model):
    bounds = np.asarray(model.bounds)
    return np.clip(t + .08*np.sin(np.arange(1, model.p+1)), bounds[:, 0]+.001, bounds[:, 1]-.001)


def target_vector(model, t, profiles, profile_weights):
    delta = np.asarray([model.common_reference(t, [1., age, female]) -
                        model.common_reference(t, [0., age, female]) for age, female in profiles])
    return np.average(delta, axis=0, weights=profile_weights)


def fit_one(run, cfg, grouped, initial, init_metadata, profiles, profile_weights):
    resolutions, t0, previous_target, chosen = [], initial.copy(), None, None
    for q, qcheck in cfg['quadrature_pairs']:
        model = Model('slope_adjusted', q, anchors=run['anchors'])
        if model.p != 60:
            raise ValueError('Unexpected anchor-model parameter count')
        fits = []
        for sid, start in enumerate((t0, perturb(t0, model)), start=1):
            tick = time.time()
            progress('fitting', model=run['id'], quadrature=q, start=sid)
            iterations = [0]
            def callback(_):
                iterations[0] += 1
                if iterations[0] % 25 == 0:
                    progress('fitting', model=run['id'], quadrature=q, start=sid, iteration=iterations[0])
            try:
                opt = minimize(model.fg, start, args=(grouped,), jac=True, method='L-BFGS-B', bounds=model.bounds,
                               options=dict(maxiter=cfg['maxiter'], ftol=cfg['ftol'], gtol=cfg['gtol'], maxls=cfg['maxls']), callback=callback)
                objective, gradient = model.fg(opt.x, grouped)
                if not (np.isfinite(objective) and np.isfinite(opt.x).all() and np.isfinite(gradient).all()):
                    raise FloatingPointError('Nonfinite optimizer parameters, objective or gradient')
                record = dict(model=run['id'], q=q, start=sid, success=bool(opt.success), message=str(opt.message),
                              iterations=int(opt.nit), evaluations=int(opt.nfev), nll=float(objective),
                              gradient_max=float(np.max(np.abs(gradient))), projected_gradient=projected(gradient, opt.x, model.bounds),
                              bound_hits=model.bounds_hits(opt.x, tolerance=cfg['boundary_tolerance']),
                              seconds=time.time()-tick, parameters=opt.x.tolist(), gradient=gradient.tolist(),
                              initial_parameters=start.tolist())
                fits.append((objective, opt.x.copy(), record))
            except Exception as exc:
                record = dict(model=run['id'], q=q, start=sid, success=False, error=repr(exc), seconds=time.time()-tick,
                              initial_parameters=start.tolist())
            ATTEMPTS.append(record)
            dump(OUT/'attempts.json', ATTEMPTS)
            progress('start_finished', **{k:v for k,v in record.items() if k not in ('parameters', 'bound_hits', 'gradient', 'initial_parameters')})
        if not fits:
            raise RuntimeError('Both optimization attempts failed at ' + run['id'] + ' q=' + str(q))
        fits.sort(key=lambda item:item[0])
        objective, t, selected = fits[0]
        check_model = Model('slope_adjusted', qcheck, anchors=run['anchors'])
        check_nll, check_gradient = check_model.fg(t, grouped)
        delta = target_vector(model, t, profiles, profile_weights)
        gap = None if len(fits) != 2 else float(max(f[0] for f in fits)-min(f[0] for f in fits))
        resolution = dict(q=q, check_q=qcheck, selected_start=selected['start'], nll=float(objective), check_nll=float(check_nll),
                          check_nll_difference=float(abs(check_nll-objective)), check_gradient_max=float(np.max(np.abs(check_gradient))),
                          multistart_gap=gap, selected_optimizer_success=selected['success'], projected_gradient=selected['projected_gradient'],
                          diagnostic_target_all8=delta.tolist(), diagnostic_target_change_from_previous_q=None if previous_target is None else (delta-previous_target).tolist())
        resolution['quadrature_pass'] = bool(resolution['check_nll_difference'] <= cfg['quadrature_nll_tolerance'] and
                                             resolution['check_gradient_max'] <= cfg['quadrature_gradient_tolerance'])
        resolution['multistart_pass'] = bool(len(fits) == 2 and gap <= cfg['multistart_nll_tolerance'] and
                                             all(f[2]['success'] and f[2]['projected_gradient'] <= cfg['projected_gradient_tolerance'] for f in fits))
        resolutions.append(resolution)
        chosen = model, check_model, t, selected, resolution
        dump(OUT/(run['id']+'_resolutions.json'), resolutions)
        if resolution['quadrature_pass']:
            break
        t0, previous_target = t.copy(), delta.copy()
    model, check_model, t, selected, resolution = chosen
    progress('hessian_check', model=run['id'], quadrature=check_model.q)
    hessian = np.empty((model.p, model.p))
    for j in range(model.p):
        step = 1e-4*max(1., abs(t[j]))
        hi, lo = t.copy(), t.copy()
        hi[j] += step
        lo[j] -= step
        hessian[:, j] = (check_model.fg(hi, grouped)[1]-check_model.fg(lo, grouped)[1])/(2*step)
    asymmetry = float(np.max(np.abs(hessian-hessian.T)))
    eigen = np.linalg.eigvalsh((hessian+hessian.T)/2)
    progress('reference_integration_check', model=run['id'])
    integration_error, coefficients = 0., model.to_components(t)
    for age, female in profiles:
        for hearing in (0., 1.):
            x = np.array([hearing, age, female])
            b = coefficients['b'] + coefficients['delta'] @ x
            a = np.exp(coefficients['loga'] + coefficients['omega'] @ x)
            numeric, _ = quad_vec(lambda z: ndtr(b+a*z)*np.exp(-z*z/2)/np.sqrt(2*np.pi), -10, 10, epsabs=1e-10, epsrel=1e-10)
            integration_error = max(integration_error, float(np.max(np.abs(numeric-model.common_reference(t,x)))))
    hits = model.bounds_hits(t, tolerance=cfg['boundary_tolerance'])
    checks = dict(optimizer=bool(selected['success'] and selected['projected_gradient'] <= cfg['projected_gradient_tolerance']),
                  multistart=resolution['multistart_pass'], quadrature=resolution['quadrature_pass'], no_bound_hits=not hits,
                  positive_observed_hessian=bool(eigen[0] > cfg['hessian_min_eigenvalue']),
                  reference_integration=bool(integration_error <= cfg['reference_integration_tolerance']))
    summary = dict(model=run['id'], stage='slope_adjusted', released_item='i'+str(run['released']+1),
                   anchors=['i'+str(j+1) for j in model.anchors], nonanchors=['i'+str(j+1) for j in model.nonanchors],
                   parameters=model.p, fit_q=model.q, check_q=check_model.q, nll=selected['nll'],
                   checks=checks, numerically_admissible=all(checks.values()), bound_hits=hits,
                   hessian_min_eigenvalue=float(eigen[0]), hessian_max_eigenvalue=float(eigen[-1]),
                   hessian_condition_number=float(eigen[-1]/eigen[0]) if eigen[0]>0 else None,
                   hessian_asymmetry_max=asymmetry, hessian_eigenvalues=eigen.tolist(),
                   hessian_symmetric=((hessian+hessian.T)/2).tolist(), parameter_names=list(model.names),
                   reference_integration_error=integration_error, check_gradient_max=resolution['check_gradient_max'],
                   check_nll_difference=resolution['check_nll_difference'], multistart_gap=resolution['multistart_gap'],
                   resolutions=resolutions, initialization=init_metadata, parameter_values=dict(zip(model.names,t.tolist())),
                   diagnostic_reference_contrasts=model.reference_contrasts(t,profiles,profile_weights,targets=model.nonanchors),
                   interpretation='Limited assumption-release sensitivity; no intervals, anchor validation, best-anchor selection or bias decomposition. Cross-anchor calibration can change.')
    dump(OUT/(run['id']+'_fit.json'), summary)
    progress('model_finished', model=run['id'], numerically_admissible=summary['numerically_admissible'], checks=checks)
    return model, check_model, t, summary


def scientific_contrasts(model, t, summary, profiles, profile_weights):
    rows = model.reference_contrasts(t, profiles, profile_weights, targets=model.nonanchors)
    for row in rows:
        row.update(model=summary['model'], released_item=summary['released_item'],
                   original_target=row['item'] in ('i5','i6','i7','i8'), numerically_admissible=summary['numerically_admissible'])
        if not summary['numerically_admissible']:
            row['contrast'], row['contrast_pp'] = None, None
    return rows


def local_diagnostics(model, t, summary, x, y, w):
    rows = []
    for hearing in (0,1):
        take = x[:,0] == hearing
        xx, yy, ww = x[take], y[take], w[take]
        observed_mean = np.average(yy, axis=0, weights=ww)
        observed_pair = (yy*ww[:,None]).T @ yy/ww.sum()
        unique_x, inverse = np.unique(xx, axis=0, return_inverse=True)
        profile_weights = np.bincount(inverse, weights=ww)
        predicted_mean, predicted_pair = np.zeros(8), np.zeros((8,8))
        for profile, weight in zip(unique_x, profile_weights):
            means, pairs = model.moments(t,profile)
            predicted_mean += weight*means/ww.sum()
            predicted_pair += weight*pairs/ww.sum()
        base = dict(model=summary['model'], group='O'+str(hearing), numerically_admissible=summary['numerically_admissible'])
        for j in range(8):
            rows.append(dict(base, kind='item', item1='i'+str(j+1), item2='', observed=observed_mean[j],
                             predicted=predicted_mean[j], residual=observed_mean[j]-predicted_mean[j]))
            for k in range(j+1,8):
                rows.append(dict(base, kind='pair', item1='i'+str(j+1), item2='i'+str(k+1), observed=observed_pair[j,k],
                                 predicted=predicted_pair[j,k], residual=observed_pair[j,k]-predicted_pair[j,k]))
    return rows


def main():
    global RUN_STARTED, START
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError('Anchor output directory already has files; refusing repeat/overwrite')
    sys.addaudithook(guard)
    manifest = validate_freeze()
    plan = read_json(ROOT/'analysis_plan.json')
    cfg = validate_anchor_plan(plan)
    prior_dir = WORKSPACE/plan['previous_stage']
    prior_path = prior_dir/'results/model_summaries.json'
    loader_path = WORKSPACE/'work/elsa-mnlfa-covariates-2026-10-24/data_audit/audit_covariates.py'
    protected = {Path(name).resolve(): value for name,value in manifest['protected'].items()}
    for required in (prior_path, loader_path):
        if required.resolve() not in protected:
            raise RuntimeError('Required prior input absent from protected hashes: '+str(required))
    source = next(s for s in read_json(prior_path) if s['model']=='slope_adjusted')
    if not source['numerically_admissible'] or source['parameters'] != 54 or source['fit_q'] != 61:
        raise RuntimeError('Expected numerically admissible prior 61-node S4')
    prior_model = Model('slope_adjusted',61)
    prior_t = np.asarray([source['parameter_values'][name] for name in prior_model.names])
    OUT.mkdir(exist_ok=True)
    START, RUN_STARTED = time.time(), True
    progress('loading', freeze_sha256=sha(ROOT/'freeze.json'))
    spec = importlib.util.spec_from_file_location('anchor_covariate_loader',loader_path)
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    full, audit = loader.load_covariate_frame('lex_first')
    if len(full)!=plan['data']['full_frame_n'] or full.hh.nunique()!=plan['data']['full_households']:
        raise RuntimeError('Original positive-weight frame changed')
    if audit['source_sha256'] != plan['data']['source_sha256']:
        raise RuntimeError('Original source hash changed')
    domain = full.loc[full.domain.eq(1)].copy()
    if len(domain)!=plan['data']['domain_n'] or domain.hh.nunique()!=plan['data']['domain_households']:
        raise RuntimeError('Original analysis domain changed')
    x = domain[['g','age10','female']].to_numpy(float)
    y = domain[[f'i{j}' for j in range(1,9)]].to_numpy(float)
    if not np.isfinite(x).all() or not np.isin(y,[0,1]).all():
        raise RuntimeError('Incomplete covariates or nonbinary items; no extra exclusions allowed')
    w = domain.w.to_numpy(dtype=float, copy=True)
    if not np.isfinite(w).all() or not np.all(w>0):
        raise RuntimeError('Original positive-weight rule changed')
    for hearing in (0,1):
        take = x[:,0]==hearing
        if int(take.sum())!=plan['data']['groups']['O'+str(hearing)]:
            raise RuntimeError('Original hearing group changed')
        w[take] *= take.sum()/w[take].sum()
    profiles, inverse = np.unique(x[:,1:],axis=0,return_inverse=True)
    profile_weights = np.bincount(inverse,weights=w)
    grouped = group_rows(y,x,w)
    summaries, contrasts, comparisons, local, parameters = [], [], [], [], []
    baseline = {r['item']:r for r in prior_model.reference_contrasts(prior_t,profiles,profile_weights)}
    for run in cfg['runs']:
        model = Model('slope_adjusted',61,anchors=run['anchors'])
        initial, metadata = model.transfer_start(prior_t,prior_model)
        if model.p!=60 or len(metadata['new_zero_names'])!=6 or metadata['dropped_names']:
            raise RuntimeError('Unexpected single-anchor release initialization')
        metadata['prior_s4_results_sha256'] = sha(prior_path)
        model, check_model, t, summary = fit_one(run,cfg,grouped,initial,metadata,profiles,profile_weights)
        summaries.append(summary)
        target_rows = scientific_contrasts(model,t,summary,profiles,profile_weights)
        contrasts.extend(target_rows)
        for row in target_rows:
            if row['original_target']:
                base = baseline[row['item']]['contrast_pp']
                comparisons.append(dict(model=run['id'],item=row['item'],prior_S4_contrast_pp=base,
                                        released_anchor_contrast_pp=row['contrast_pp'],
                                        descriptive_change_pp=None if row['contrast_pp'] is None else row['contrast_pp']-base,
                                        numerically_admissible=summary['numerically_admissible'],
                                        interpretation='Cross-anchor model/reference-conditional descriptive change; not eliminated DIF or causal bias'))
        parameters.extend(dict(model=run['id'],parameter=name,value=value) for name,value in zip(model.names,t))
        local.extend(local_diagnostics(check_model,t,summary,x,y,w))
        pd.DataFrame(contrasts).to_csv(OUT/'reference_contrasts.csv',index=False)
        pd.DataFrame(comparisons).to_csv(OUT/'original_target_comparison.csv',index=False)
        pd.DataFrame(parameters).to_csv(OUT/'parameters.csv',index=False)
        pd.DataFrame(local).to_csv(OUT/'local_fit.csv',index=False)
        dump(OUT/'model_summaries.json',summaries)
    validate_freeze()
    completion = dict(completed_utc=stamp(),seconds=time.time()-START,n=len(domain),households=int(domain.hh.nunique()),
                      groups=plan['data']['groups'],four_anchor_models_complete=len(summaries)==4,
                      numerically_admissible=sum(s['numerically_admissible'] for s in summaries),
                      freeze_sha256=sha(ROOT/'freeze.json'),protected_sources_unchanged=True,
                      python=sys.version,numpy=np.__version__,scipy=scipy.__version__,pandas=pd.__version__,
                      original_targets=4,free_targets_per_model=5,models=len(summaries),profiles_exported=False,
                      inference='Exploratory one-at-a-time anchor assumption release only; no intervals, p values, anchor validation or causal claims.',
                      participant_rows_exported=0,pattern_scores_exported=0,household_scores_exported=0,
                      output_hashes={p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='progress.json'})
    dump(OUT/'completion.json',completion)
    progress('completed',models=len(summaries),numerically_admissible=completion['numerically_admissible'])


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        if RUN_STARTED:
            progress('failed',error=repr(exc))
        raise
