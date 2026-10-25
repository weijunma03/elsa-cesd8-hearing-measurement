"""Full positive-weight W7 design frame; individual records stay in RAM/pipes.

Public callable: full_frame_from_original(tie_rule='lex_first') -> (frame,audit).
Only --stream prints rows, solely to a local pipe consumed by design_adapter.R.
Default CLI writes aggregate checks, never participant records or original IDs.
"""
from pathlib import Path
import hashlib
import io
import importlib.util
import json
import sys
import zipfile
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OLD = ROOT / 'work/elsa-anchor-structure-2026-10-21'
EXPECTED_HASH = '181c00df9becc0b8f01752db20a0569f6c4afa115172ed944a1e0dec47c52294'
ITEMS = ['psced' + c for c in 'abcdefgh']

def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()

def integer(s, lo, hi):
    return s.notna() & s.between(lo, hi) & s.eq(s.round())

def modal(s, last=False):
    freq = s.astype(str).value_counts()
    return sorted(freq[freq == freq.max()].index)[-1 if last else 0]

def full_frame_from_original(tie_rule='lex_first'):
    if tie_rule not in ('lex_first', 'lex_last'):
        raise ValueError('Only predeclared lex_first/lex_last tie rules are allowed')
    source = json.loads((ROOT/'work/elsa-intake-2026-10-20/integrity.json').read_text(encoding='utf-8'))
    old = json.loads((OLD/'manifest.json').read_text(encoding='utf-8'))
    cfg = json.loads((ROOT/'work/elsa-feasibility-2026-10-20/analysis_field_metadata.json').read_text(encoding='utf-8'))['7']
    assert sha(source['archive']) == source['sha256'] == old['source_sha256'] == EXPECTED_HASH
    assert cfg['member'] == old['source_member']
    fields = ['idauniq','samptyp','finstatw7','indager','askpx','w7indout','idahhw7','hehlfa','hehlfb','hehrga','hehrgb','hehlfn','hehrgn','w7xwgt','gor'] + ITEMS
    with zipfile.ZipFile(source['archive']) as archive:
        b = pd.read_stata(io.BytesIO(archive.read(cfg['member'])), columns=fields,
                          convert_categoricals=False, preserve_dtypes=False)
    b.columns = [str(c).lower() for c in b.columns]
    assert len(b) == 9670 and b.idauniq.notna().all() and b.idauniq.is_unique
    base = (integer(b.indager,50,89) | b.indager.eq(-7)) & b.samptyp.eq(1) & b.askpx.eq(0) & b.w7indout.isin([11,21,24])
    assert base.sum() == 7894 and not (base & ~b.finstatw7.isin([1,7,14,25,33])).any()
    components = pd.concat([integer(b[c],0,3) for c in ['hehlfa','hehlfb','hehrga','hehrgb']], axis=1).all(axis=1)
    screen = components & integer(b.hehlfn,0,6) & integer(b.hehrgn,0,6) & b.hehlfn.eq(b.hehlfa+b.hehlfb) & b.hehrgn.eq(b.hehrga+b.hehrgb)
    w = pd.to_numeric(b.w7xwgt,errors='coerce')
    gor_num = pd.to_numeric(b.gor,errors='coerce')
    gor_text = b.gor.astype('string').str.strip()
    gor_ok = integer(gor_num,1,9) | gor_text.str.fullmatch(r'E1200000[1-9]|W99999999|S99999999|N99999999',na=False)
    selected = base & screen & b[ITEMS].isin([1,2]).all(axis=1) & integer(b.idahhw7,1,10**12) & np.isfinite(w) & w.gt(0) & gor_ok
    positive = np.isfinite(w) & w.gt(0)
    assert int(positive.sum()) == 8152 and int(selected.sum()) == 7314
    assert integer(b.loc[positive,'idahhw7'],1,10**12).all() and gor_ok[positive].all()
    # Reproduce the OLD household split using only its original final-domain rows.
    old_selected = pd.DataFrame({'hh':b.loc[selected,'idahhw7'].astype('int64'),
                                 'region':gor_text[selected].astype(str)})
    split_hh = old_selected.groupby('hh',sort=True).region.agg(modal).rename('split_region').reset_index().sort_values(['split_region','hh'])
    rng = np.random.default_rng(20261020)
    partition_map = {}
    for _, sub in split_hh.groupby('split_region',sort=True):
        ids = sub.hh.to_numpy(copy=True); rng.shuffle(ids)
        nd = max(1,min(len(ids)-1,int(round(.6*len(ids)))))
        partition_map.update({int(h):'discovery' for h in ids[:nd]})
        partition_map.update({int(h):'confirmation' for h in ids[nd:]})
    full = b.loc[positive,['idauniq','idahhw7']].copy()
    full['w'] = w[positive].astype(float)
    full['region_raw'] = gor_text[positive].astype(str)
    full['domain'] = selected[positive].astype(int)
    full['partition'] = full.idahhw7.map(partition_map).fillna('')
    representatives = full.groupby('idahhw7',sort=True).region_raw.agg(lambda s:modal(s,last=tie_rule=='lex_last'))
    full['stratum'] = full.idahhw7.map(representatives)
    objective = pd.Series(np.where(np.maximum(b.hehlfb,b.hehrgb)<3,'O1','O0'),index=b.index)
    full['O'] = objective[positive].where(full.domain.eq(1), '')
    for i,c in enumerate(ITEMS,1):
        full['i'+str(i)] = b.loc[positive,c].eq(1 if c in ['psced'+k for k in 'abcegh'] else 2).astype(float).where(full.domain.eq(1))
    group_counts = full.loc[full.domain.eq(1)].groupby(['partition','O']).agg(n=('O','size'),households=('idahhw7','nunique'),weight_sum=('w','sum')).reset_index()
    pd.testing.assert_frame_equal(group_counts,pd.DataFrame(old['partition_group_counts']),check_dtype=False,check_exact=False,rtol=1e-12)
    # Opaque stable codes are formed from the entire positive frame, not the domain.
    hh_map = {h:f'H{k:06d}' for k,h in enumerate(sorted(full.idahhw7.unique()),1)}
    person_map = {p:f'P{k:06d}' for k,p in enumerate(sorted(full.idauniq.unique()),1)}
    full['hh'] = full.idahhw7.map(hh_map)
    full['person_id'] = full.idauniq.map(person_map)
    frame = full[['person_id','hh','stratum','w','domain','partition','O']+['i'+str(i) for i in range(1,9)]].reset_index(drop=True)
    n_by_stratum = frame[['hh','stratum']].drop_duplicates().groupby('stratum').size()
    audit = {'source_sha256':EXPECTED_HASH,'source_member':cfg['member'],
        'tie_rule':tie_rule,'frozen_split_rule':'Original final-domain modal GOR, lexical-first tie; NumPy seed 20261020; unchanged for all variance sensitivity rules',
        'positive_frame_people':len(frame),'positive_frame_households':frame.hh.nunique(),
        'domain_people':int(frame.domain.sum()),'domain_households':frame.loc[frame.domain.eq(1),'hh'].nunique(),
        'zero_domain_households':int(frame.groupby('hh').domain.sum().eq(0).sum()),
        'excluded_only_persons':int(frame.partition.eq('').sum()),
        'excluded_people_in_domain_households':int((frame.domain.eq(0)&frame.partition.ne('')).sum()),
        'gor_conflict_households':int(full.groupby('idahhw7').region_raw.nunique().gt(1).sum()),
        'strata':len(n_by_stratum),'households_by_stratum':{str(k):int(v) for k,v in n_by_stratum.items()},
        'design_df':int(frame.hh.nunique()-len(n_by_stratum)),
        'partition_group_counts':group_counts.to_dict('records'),
        'participant_files_written':0,'models_fitted':0,'original_identifiers_leave_loader':False}
    return frame,audit

def main():
    if '--stream' in sys.argv:
        rule = sys.argv[sys.argv.index('--stream')+1] if len(sys.argv)>sys.argv.index('--stream')+1 else 'lex_first'
        frame,_ = full_frame_from_original(rule)
        frame.to_csv(sys.stdout,index=False,lineterminator='\n',na_rep='')
        return
    frame,audit = full_frame_from_original()
    alternate,alt_audit = full_frame_from_original('lex_last')
    checks = {}
    def check(name,value):checks[name]=bool(value)
    check('full_frame_8152_people_6217_hh',len(frame)==8152 and frame.hh.nunique()==6217)
    check('domain_7314_people_5748_hh',frame.domain.sum()==7314 and frame.loc[frame.domain.eq(1),'hh'].nunique()==5748)
    check('zero_domain_469_hh_537_people',audit['zero_domain_households']==469 and audit['excluded_only_persons']==537)
    check('301_excluded_people_in_domain_hh',audit['excluded_people_in_domain_households']==301)
    check('all_domain_members_partitioned',frame.loc[frame.domain.eq(1),'partition'].isin(['discovery','confirmation']).all())
    check('excluded_only_hh_have_no_invented_partition',frame.groupby('hh').apply(lambda d:d.partition.eq('').all() if d.domain.sum()==0 else d.partition.nunique()==1,include_groups=False).all())
    check('household_kept_whole_one_stratum',frame.groupby('hh').stratum.nunique().eq(1).all())
    check('nine_strata_no_singleton',audit['strata']==9 and min(audit['households_by_stratum'].values())>1)
    check('design_df_6208',audit['design_df']==6208)
    check('person_and_hh_codes_opaque',frame.person_id.str.fullmatch('P[0-9]{6}').all() and frame.hh.str.fullmatch('H[0-9]{6}').all())
    check('person_codes_unique',frame.person_id.is_unique)
    check('excluded_model_values_blank',frame.loc[frame.domain.eq(0),'O'].eq('').all() and frame.loc[frame.domain.eq(0),['i'+str(i) for i in range(1,9)]].isna().all().all())
    check('domain_binary_complete',frame.loc[frame.domain.eq(1),['i'+str(i) for i in range(1,9)]].isin([0,1]).all().all())
    check('all_weights_positive_finite',np.isfinite(frame.w).all() and frame.w.gt(0).all())
    check('alternative_changes_only_stratum',frame.drop(columns='stratum').equals(alternate.drop(columns='stratum')))
    change=frame.stratum.ne(alternate.stratum)
    check('alternate_changes_three_whole_two_person_hh',change.sum()==6 and frame.loc[change,'hh'].nunique()==3 and frame.loc[change].groupby('hh').size().eq(2).all())
    check('alternate_preserves_full_design_counts',alt_audit['positive_frame_people']==8152 and alt_audit['positive_frame_households']==6217 and alt_audit['design_df']==6208)
    check('partition_counts_discovery4380_confirmation2934',frame.loc[frame.domain.eq(1),'partition'].value_counts().to_dict()=={'discovery':4380,'confirmation':2934})
    # Independently call the unchanged previous loader, retaining its CSV only
    # in RAM. Compare every domain model row and the household equivalence map.
    old_loader=ROOT/'work/elsa-household-pilot-2026-10-21/loader.py'
    old_hash=sha(old_loader)
    spec=importlib.util.spec_from_file_location('previous_pilot_loader',old_loader)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    payload,_=module.load_local()
    model_columns=['O','w']+['i'+str(i) for i in range(1,9)]
    for part in ('discovery','confirmation'):
        prior=pd.read_csv(io.StringIO(payload[part]))
        now=frame.loc[frame.domain.eq(1)&frame.partition.eq(part)].reset_index(drop=True)
        values_equal=prior.O.equals(now.O) and np.allclose(prior.w,now.w,rtol=1e-14,atol=0) and np.array_equal(prior[model_columns[2:]],now[model_columns[2:]])
        check(part+'_all_model_rows_match_unchanged_previous_loader',values_equal)
        mapping=pd.DataFrame({'old':prior.hh,'new':now.hh})
        check(part+'_same_household_equivalence_and_baseline_stratum',mapping.groupby('old').new.nunique().eq(1).all() and mapping.groupby('new').old.nunique().eq(1).all() and prior.region.equals(now.stratum))
    check('previous_loader_unchanged',sha(old_loader)==old_hash=='edc0b9607354492cea44cd1d2bcd23810a2d4b96380d629d3c3e9005d76edd5e')
    audit['alternate_tie_sensitivity']={'rule':'lex_last','changed_households':3,'changed_people':6,'partition_unchanged':True,'not_official_truth':True,'households_by_stratum':alt_audit['households_by_stratum']}
    (HERE/'frame_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    (HERE/'frame_checks.json').write_text(json.dumps({'passed':sum(checks.values()),'failed':sum(not x for x in checks.values()),'checks':checks},indent=2),encoding='utf-8')
    assert all(checks.values()),checks
    print(json.dumps({'frame_checks_passed':len(checks),'person_rows_written':0,'models_fitted':0}))

if __name__=='__main__':main()
