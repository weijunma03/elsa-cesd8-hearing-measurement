"""Read-only original ELSA domain audit; export aggregate tables only.

load_covariate_frame() is a reusable local RAM-only loader. It augments the
unchanged 8,152-person positive-weight frame with age and sex and verifies
original row alignment. This module never writes participant rows or IDs.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import importlib.util
import importlib.metadata
import io
import json
import platform
import sys
import zipfile

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
LOADER = ROOT / 'work/elsa-variance-adapter-2026-10-21/design/frame_loader.py'
ITEMS = ['psced' + c for c in 'abcdefgh']
ITEM_COLS = ['i' + str(i) for i in range(1, 9)]
AGE_LABELS = ['50-59', '60-69', '70-79', '80-89', '90+']
EXPECTED_SOURCE = '181c00df9becc0b8f01752db20a0569f6c4afa115172ed944a1e0dec47c52294'


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def load_covariate_frame(tie_rule='lex_first'):
    """Return full positive-weight frame plus age/sex; no persistent row data."""
    spec = importlib.util.spec_from_file_location('frozen_design_loader', LOADER)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    frame, frame_audit = original.full_frame_from_original(tie_rule)
    source = json.loads((ROOT/'work/elsa-intake-2026-10-20/integrity.json').read_text(encoding='utf-8'))
    cfg = json.loads((ROOT/'work/elsa-feasibility-2026-10-20/analysis_field_metadata.json').read_text(encoding='utf-8'))['7']
    assert source['sha256'] == frame_audit['source_sha256'] == EXPECTED_SOURCE
    fields = ['indager', 'indsex', 'w7xwgt', 'hehlfb', 'hehrgb'] + ITEMS
    with zipfile.ZipFile(source['archive']) as z:
        raw = pd.read_stata(io.BytesIO(z.read(cfg['member'])), columns=fields,
                            convert_categoricals=False, preserve_dtypes=False)
    raw.columns = [str(c).lower() for c in raw.columns]
    pos = np.isfinite(raw.w7xwgt) & raw.w7xwgt.gt(0)
    aux = raw.loc[pos].reset_index(drop=True)
    assert len(raw) == 9670 and len(aux) == len(frame) == 8152
    assert np.array_equal(aux.w7xwgt.to_numpy(float), frame.w.to_numpy(float))
    domain = frame.domain.eq(1)
    original_groups = np.where(np.maximum(aux.hehlfb, aux.hehrgb) < 3, 'O1', 'O0')
    assert np.array_equal(original_groups[domain], frame.loc[domain, 'O'])
    # Independent item alignment check in every retained row without using IDs.
    for j, c in enumerate(ITEMS, 1):
        code = 2 if c in ['pscedd', 'pscedf'] else 1
        assert np.array_equal(aux.loc[domain, c].eq(code).to_numpy(float),
                              frame.loc[domain, f'i{j}'].to_numpy(float))
    frame = frame.copy()
    frame['age_raw'] = aux.indager
    valid_age = aux.indager.eq(-7) | (aux.indager.between(50, 89) & aux.indager.eq(aux.indager.round()))
    frame['age_capped90'] = aux.indager.mask(aux.indager.eq(-7), 90).where(valid_age)
    frame['age90plus'] = aux.indager.eq(-7).where(valid_age)
    frame['age10'] = (frame.age_capped90 - 70.0) / 10.0
    frame['sex_raw'] = aux.indsex
    frame['female'] = aux.indsex.eq(2).astype(float).where(aux.indsex.isin([1, 2]))
    frame['g'] = frame.O.map({'O0': 0.0, 'O1': 1.0})
    return frame, frame_audit


def weight_summary(data):
    w = data.w.to_numpy(float)
    age = data.age_capped90.to_numpy(float)
    return {
        'n': len(data), 'households': int(data.hh.nunique()),
        'weight_sum': float(w.sum()),
        'kish_weight_ess': float(w.sum()**2 / np.dot(w, w)) if len(data) else None,
        'age_min_capped90': float(age.min()) if len(data) else None,
        'age_max_capped90': float(age.max()) if len(data) else None,
        'age_mean_capped90': float(age.mean()) if len(data) else None,
        'weighted_age_mean_capped90': float(np.average(age, weights=w)) if len(data) else None,
        'age90plus_n': int(data.age_raw.eq(-7).sum()),
        'female_n': int(data.female.eq(1).sum()),
        'female_percent': float(100*data.female.mean()) if len(data) else None,
    }


def main():
    if (HERE/'audit.json').exists():
        raise RuntimeError('Completed audit exists; no silent overwrite')
    # Local-only execution even though numpy/pandas are unrestricted libraries.
    network_events = []
    def guard(event, args):
        if event in {'socket.connect', 'socket.connect_ex', 'socket.getaddrinfo', 'socket.sendto'}:
            network_events.append(event)
            raise RuntimeError('Participant workflow is local only')
    sys.addaudithook(guard)
    protected = [LOADER,
        ROOT/'work/elsa-final-sample-2026-10-21/manifest.json',
        ROOT/'outputs/elsa-measurement-manuscript-2026-10-24.docx',
        ROOT/'work/elsa-real-analysis-2026-10-24/results/fits.csv']
    before = {str(p.relative_to(ROOT)): sha(p) for p in protected}
    frame, base = load_covariate_frame()
    d = frame.loc[frame.domain.eq(1)].copy()
    checks = {}
    def check(name, value):
        checks[name] = bool(value)
        if not checks[name]:
            raise AssertionError(name)
    check('domain_n_7314', len(d) == 7314)
    check('domain_households_5748', d.hh.nunique() == 5748)
    check('group_counts_unchanged', d.O.value_counts().to_dict() == {'O0': 5241, 'O1': 2073})
    check('age_complete_valid', d.age_capped90.notna().all())
    check('sex_complete_valid', d.female.notna().all())
    check('female_count_4109', d.female.eq(1).sum() == 4109)
    check('complete_binary_items', d[ITEM_COLS].isin([0, 1]).all().all())
    check('positive_weights', np.isfinite(d.w).all() and d.w.gt(0).all())
    check('90plus_mapped_only_from_minus7', np.array_equal(d.age_raw.eq(-7), d.age_capped90.eq(90)))
    check('age10_encoding', np.array_equal((d.age_capped90-70)/10, d.age10))
    d['sex'] = d.female.map({0.0: 'Male', 1.0: 'Female'})
    d['age_band'] = pd.cut(d.age_capped90, [49, 59, 69, 79, 89, 90], labels=AGE_LABELS)
    groups = []
    for group in ['all', 'O0', 'O1']:
        sub = d if group == 'all' else d.loc[d.O.eq(group)]
        groups.append({'group': group, **weight_summary(sub)})
    pd.DataFrame(groups).to_csv(HERE/'group_summary.csv', index=False, encoding='utf-8-sig')
    cells = []
    totals = []
    for group in ['O0', 'O1']:
        for sex in ['Male', 'Female']:
            sub = d.loc[d.O.eq(group) & d.sex.eq(sex)]
            totals.append({'group': group, 'sex': sex, **weight_summary(sub)})
            for band in AGE_LABELS:
                cell = sub.loc[sub.age_band.eq(band)]
                stat = weight_summary(cell)
                stat['percent_within_group_sex'] = 100 * len(cell)/len(sub)
                stat['weighted_percent_within_group_sex'] = 100*cell.w.sum()/sub.w.sum()
                cells.append({'group': group, 'sex': sex, 'age_band': band, **stat})
    pd.DataFrame(totals).to_csv(HERE/'group_sex_summary.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(cells).to_csv(HERE/'group_sex_age_band_summary.csv', index=False, encoding='utf-8-sig')
    check('age_band_cells_sum_7314', sum(r['n'] for r in cells) == 7314)
    check('all20_predefined_age_sex_group_cells_nonempty', all(r['n'] > 0 for r in cells))
    # This is descriptive overlap of recorded covariates, not propensity overlap.
    support = []
    for sex in ['Male', 'Female']:
        a0 = d.loc[d.O.eq('O0') & d.sex.eq(sex), 'age_capped90']
        a1 = d.loc[d.O.eq('O1') & d.sex.eq(sex), 'age_capped90']
        common_low = float(max(a0.min(), a1.min()))
        common_high = float(min(a0.max(), a1.max()))
        exact_age_counts = d.loc[d.sex.eq(sex)].groupby(['age_capped90', 'O'], observed=True).size().unstack(fill_value=0)
        joint_ages = exact_age_counts.index[(exact_age_counts['O0'] > 0) & (exact_age_counts['O1'] > 0)].to_numpy(float)
        support.append({'sex': sex, 'common_recorded_age_min': common_low,
                        'common_recorded_age_max_capped90': common_high,
                        'recorded_age_levels_both_groups_present': len(joint_ages),
                        'within_common_range_n': int(d.loc[d.sex.eq(sex), 'age_capped90'].between(common_low, common_high).sum()),
                        'outside_common_range_n': int((~d.loc[d.sex.eq(sex), 'age_capped90'].between(common_low, common_high)).sum()),
                        'minimum_n_in_group_exact_recorded_age_cell': int(exact_age_counts.min().min()),
                        'note': '90 is an open-ended 90+ category; overlapping ranges do not imply balanced covariates.'})
    pd.DataFrame(support).to_csv(HERE/'common_support_summary.csv', index=False, encoding='utf-8-sig')
    # Basic finite linear design checks, not a fitted outcome/propensity model.
    X = np.column_stack([np.ones(len(d)), d.g, d.age10, d.female])
    singular = np.linalg.svd(X, compute_uv=False)
    rank = int(np.linalg.matrix_rank(X))
    check('fixed_covariate_design_full_rank', rank == 4)
    runtime = {'python': platform.python_version(), 'executable': sys.executable,
               'numpy_version': np.__version__, 'pandas_version': pd.__version__, 'modules': {}}
    for module in ['numpy', 'scipy', 'torch', 'jax', 'jaxlib', 'numba']:
        try:
            found = importlib.util.find_spec(module) is not None
        except (ValueError, ImportError):
            found = False
        try:
            version = importlib.metadata.version(module) if found else None
        except importlib.metadata.PackageNotFoundError:
            version = None
        runtime['modules'][module] = {'find_spec_available': found, 'distribution_version': version}
    after = {str(p.relative_to(ROOT)): sha(p) for p in protected}
    check('protected_sources_unchanged', before == after)
    check('no_network_events', len(network_events) == 0)
    result = {
        'completed_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Covariate feasibility and common-support audit only; no models, tests or participant exports',
        'source_sha256': EXPECTED_SOURCE, 'source_member': base['source_member'],
        'loader_sha256': sha(LOADER), 'script_sha256': sha(__file__),
        'parent_frame_counts': base, 'sample_n': len(d), 'groups': groups,
        'covariate_missing': {'age': int(d.age_capped90.isna().sum()), 'sex': int(d.female.isna().sum())},
        'encoding': {
            'g': '0=O0, 1=O1; original 3kHz screening grouping unchanged',
            'age_capped90': 'indager 50..89 retained; raw -7 mapped to 90 for numeric summaries and model input, explicitly top-coded',
            'age10': '(age_capped90 - 70) / 10',
            'female': 'indsex 1=male->0, 2=female->1; all other codes missing',
            'age_bands': AGE_LABELS,
            'age_band_note': 'Five prespecified categories; first four are 10-year bands, final category is 90+. Not five-year intervals.',
            'new_exclusions': 0,
        },
        'common_support': support,
        'minimum_n_in_20_predeclared_group_sex_age_band_cells': min(r['n'] for r in cells),
        'linear_covariate_matrix': {'columns': ['intercept', 'g', 'age10', 'female'], 'rank': rank,
                                    'condition_number': float(singular[0]/singular[-1])},
        'limitations': [
            'Age ranges overlap, but age and sex composition can still differ markedly between hearing groups.',
            'The top-coded 90+ category is not exact age; later nonlinearity/top-code sensitivity must be explicitly specified.',
            'Kish ESS reflects only unequal weights; it is not a survey design-adjusted effective sample size.',
            'No demographic adjustment or causal effect has yet been estimated in this audit.',
            'Keeping the original domain avoids a new covariate-completeness exclusion, but does not resolve original complete-case selection.',
        ],
        'runtime': runtime, 'checks': checks, 'checks_passed': len(checks),
        'passed': all(checks.values()), 'protected_hashes': before,
        'participant_rows_exported': 0, 'participant_ids_exported': 0,
        'models_fitted': 0, 'p_values_computed': 0, 'network_events': len(network_events),
    }
    (HERE/'audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'passed': result['passed'], 'checks': len(checks), 'sample_n': len(d),
                      'groups': groups, 'missing': result['covariate_missing'],
                      'minimum_cell_n': result['minimum_n_in_20_predeclared_group_sex_age_band_cells'],
                      'runtime_modules': runtime['modules']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
