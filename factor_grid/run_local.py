"""Read authorized W7 locally, pipe to R, save aggregate outputs only."""
import hashlib,importlib.util,json,os,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,o):p.write_text(json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
network_events=[]
def guard(event,args):
    if event in {'socket.connect','socket.connect_ex','socket.getaddrinfo','socket.sendto'}:
        network_events.append(event);raise RuntimeError('Participant workflow is local only')
sys.addaudithook(guard)
plan=json.loads((HERE/'analysis_plan.json').read_text(encoding='utf-8'))
freeze=json.loads((HERE/'plan_freeze.json').read_text(encoding='utf-8'))
assert sha(HERE/'analysis_plan.json')==freeze['plan_sha256']
assert json.loads((HERE/'review/execution_review.json').read_text(encoding='utf-8'))['passed'] is True
for p,h in plan['source_hashes'].items():assert sha(ROOT/p)==h,p
assert not (HERE/'launch_record.json').exists(),'No repeated launch'
write(HERE/'launch_record.json',{'started_utc':datetime.now(timezone.utc).isoformat(),'plan_sha256':freeze['plan_sha256'],'planned_fits':60,'pid':os.getpid()})
lp=ROOT/'work/elsa-variance-adapter-2026-10-21/design/frame_loader.py'
spec=importlib.util.spec_from_file_location('frozen_frame_loader',lp);loader=importlib.util.module_from_spec(spec);spec.loader.exec_module(loader)
f,a=loader.full_frame_from_original('lex_first');alt,aa=loader.full_frame_from_original('lex_last')
assert f.drop(columns='stratum').equals(alt.drop(columns='stratum'))
assert int(f.stratum.ne(alt.stratum).sum())==6
data=f.rename(columns={'stratum':'region'}).copy();data['region_alt']=alt.stratum
cols=['hh','region','region_alt','w','domain','partition','O']+['i'+str(i) for i in range(1,9)]
data=data[cols]
assert len(data)==8152 and int(data.domain.sum())==7314
def native(o):
    if isinstance(o,dict):return {k:native(v) for k,v in o.items()}
    if isinstance(o,list):return [native(v) for v in o]
    if isinstance(o,np.generic):return o.item()
    return o
write(HERE/'input_audit.json',native({'primary':a,'alternative':aa,'changed_people':6,'original_identifiers_sent_to_R':False,'person_level_disk_outputs':0}))
env=os.environ.copy()
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:env[key]='1'
with (HERE/'run_stdout.txt').open('w',encoding='utf-8') as out,(HERE/'run_stderr.txt').open('w',encoding='utf-8') as err:
    completed=subprocess.run([r'C:\Program Files\R\R-4.6.1\bin\Rscript.exe','--vanilla',str(HERE/'run_analysis.R')],
       input=data.to_csv(index=False,lineterminator='\n',na_rep=''),text=True,encoding='utf-8',stdout=out,stderr=err,cwd=ROOT,env=env,
       creationflags=subprocess.CREATE_NO_WINDOW)
del data,f,alt
write(HERE/'process_exit.json',{'exit_code':completed.returncode,'finished_utc':datetime.now(timezone.utc).isoformat(),'network_attempts':len(network_events)})
if completed.returncode:raise RuntimeError('R stage failed; preserve journals, no automatic retry')
assert not network_events
for p,h in plan['source_hashes'].items():assert sha(ROOT/p)==h,p
c=json.loads((HERE/'results/completion.json').read_text(encoding='utf-8'))
assert c['attempted_fits']==60 and c['status']=='complete'
write(HERE/'execution_complete.json',{'status':'complete_pending_independent_audit','completed_utc':datetime.now(timezone.utc).isoformat(),
  'planned_and_attempted_model_fits':60,'input_sources_unchanged':True,'participant_files_written':0,'network_attempts':0,
  'output_hashes':{str(p.relative_to(HERE)):sha(p) for p in sorted((HERE/'results').iterdir()) if p.is_file()}})
print(json.dumps({'complete':True,'fits':60,'fit_successes':c['fit_successes'],'reference_interval_available':c['reference_interval_available']}))
