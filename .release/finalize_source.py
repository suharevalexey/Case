"""Apply the final, audited source changes to the main-derived release payload."""
from pathlib import Path
import textwrap
ROOT=Path(__file__).resolve().parents[1]
def write(name,text):
    path=ROOT/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_text(textwrap.dedent(text).lstrip(),encoding='utf-8')
def replace(name,before,after):
    path=ROOT/name; text=path.read_text()
    if before not in text: raise ValueError('Unexpected source at '+name+': '+repr(before[:70]))
    path.write_text(text.replace(before,after),encoding='utf-8')

for prop,direction in {'absorption_max_nm':'range','log_extinction':'maximize','pss_fraction':'maximize','log_half_life':'maximize','log_kp':'minimize','skin_sensitization':'minimize','skin_irritation':'minimize'}.items():
    replace('case/common.py',f" '{prop}': dict(",f" '{prop}': dict(objective='{direction}', ")
replace('case/common.py','def features(smiles: str):\n    mol =',"def features(smiles: str):\n    if not isinstance(smiles,str) or not smiles.strip(): return None\n    mol =")
replace('case/common.py','def passes(values, spec):', '''def utility_margin(values, spec):
    """Physical bounds validate predictions; they are not competing objectives."""
    values=np.asarray(values,dtype=float)
    if spec['objective']=='maximize': return values-spec['low']
    if spec['objective']=='minimize': return spec['high']-values
    return margin(values,spec)


def passes(values, spec):''')

write('case/reward.py','''
"""B0 scalarization and M1 exact front-preserving, batch-dependent reward."""
from __future__ import annotations
import numpy as np
from .common import TARGETS,BETA,AD_LIMIT,SA_LIMIT,utility_margin


def ranks(utilities):
    u=np.asarray(utilities,float)
    if u.ndim!=2 or not np.isfinite(u).all(): raise ValueError('Finite 2-D utilities required')
    n=len(u); rank=np.full(n,-1,int)
    if not n: return rank
    dominates=np.all(u[:,None,:]>=u[None,:,:],axis=2)&np.any(u[:,None,:]>u[None,:,:],axis=2)
    remaining=np.ones(n,bool); level=0
    while remaining.any():
        idx=np.flatnonzero(remaining); front=idx[~dominates[np.ix_(idx,idx)].any(axis=0)]
        if not len(front): raise RuntimeError('Invalid dominance graph')
        rank[front]=level; remaining[front]=False; level+=1
    return rank


def front_reward(rank,tie):
    rank=np.asarray(rank,float); tie=np.asarray(tie,float)
    if not np.isfinite(rank).all() or not np.isfinite(tie).all(): raise ValueError('Finite ranks and ties required')
    if np.any(rank<0) or np.any(rank!=np.floor(rank)) or np.any((tie<0)|(tie>1)):
        raise ValueError('Nonnegative integer ranks and ties in [0,1] required')
    return 1/(1+rank+.5*(1-tie))


def score(frame,method,priority=0,beta=BETA,use_ad=True):
    if method not in {'B0','M1'}: raise ValueError('Only B0/M1')
    if not np.isfinite(beta) or beta<0: raise ValueError('Finite nonnegative penalty required')
    if not isinstance(priority,(int,np.integer)) or not 0<=priority<len(TARGETS): raise ValueError('Unknown priority')
    result=np.zeros(len(frame),float)
    if not len(frame) or not frame.valid.any(): return result
    eligible=frame.valid.to_numpy(dtype=bool,copy=True)
    sa=frame.synthetic_accessibility.to_numpy(float)
    eligible &= np.isfinite(sa)&(sa<=SA_LIMIT)
    required=[prefix+p for p in TARGETS for prefix in ['pred_','unc_']]
    eligible &= np.isfinite(frame[required].to_numpy(float)).all(axis=1)
    eligible &= (frame[['unc_'+p for p in TARGETS]].to_numpy(float)>=0).all(axis=1)
    if use_ad:
        distances=frame[['dist_to_D_A','dist_to_D_B']].to_numpy(float)
        eligible &= np.isfinite(distances).all(axis=1)&(distances>=0).all(axis=1)&(distances<=1).all(axis=1)
    idx=np.flatnonzero(eligible)
    if not len(idx): return result
    selected=frame.iloc[idx]; utilities=[]
    for prop,spec in TARGETS.items():
        values=selected['pred_'+prop].to_numpy(float)
        spread=selected['unc_'+prop].to_numpy(float)
        z=(utility_margin(values,spec)-beta*spread)/spec['scale']
        utilities.append(1/(1+np.exp(-np.clip(z,-60,60))))
    u=np.stack(utilities,axis=1)
    if use_ad:
        worst=selected[['dist_to_D_A','dist_to_D_B']].max(axis=1).to_numpy(float)
        u *= np.exp(-10*np.maximum(worst-AD_LIMIT,0))[:,None]
    gm=np.exp(np.mean(np.log(np.clip(u,1e-15,1)),axis=1))
    result[idx]=gm if method=='B0' else front_reward(ranks(u),.7*u[:,priority]+.3*gm)
    return result
''')

replace('case/models.py','from sklearn.model_selection import GroupShuffleSplit\n','')
replace('case/models.py',"frame=pd.read_csv(path); X,valid=cached_features(path,path.with_name('features.npz'))", "from .data import verify_processed\n    verify_processed(root)\n    frame=pd.read_csv(path); X,valid=cached_features(path,path.with_name('features.npz'))")
replace('case/models.py',"oracle_scope='Cross-model audit proxy trained on same labels as guidance; not experimental truth', models={})", "oracle_scope='Cross-model audit proxy trained on same labels as guidance; not experimental truth',\n                  training_code_sha256={name:sha256(ROOT/'case'/name) for name in ['common.py','data.py','models.py']}, models={})")
replace('case/models.py',"if self.manifest['targets']!=TARGETS: raise ValueError('Target definition mismatch; retrain')", "if self.manifest['targets']!=TARGETS: raise ValueError('Target definition mismatch; retrain')\n        if set(self.manifest['models'])!=set(TARGETS): raise ValueError('Incomplete model registry')\n        if sha256(self.directory/'splits.csv')!=self.manifest['splits_sha256']: raise ValueError('Split manifest checksum mismatch')")
replace('case/models.py',"for key in ['rdkit','scikit-learn','xgboost','numpy']:","for key in ['rdkit','scikit-learn','xgboost','numpy','joblib']:")
replace('case/models.py',"if method_id is not None: result['method_id']=method_id", '''result['final_evaluator_used']=self.include_oracle
        result['status']='invalid'
        for name in ['synthetic_accessibility','dist_to_D_A','dist_to_D_B']:
            result[name]=np.nan
        for prop,spec in TARGETS.items():
            for prefix in ['pred_','unc_','oracle_','dist_']: result[prefix+prop]=np.nan
            for role in ['guidance','oracle']:
                if spec['kind']=='classification':
                    result[f'{role}_{prop}_set0']=False; result[f'{role}_{prop}_set1']=False
                else:
                    result[f'{role}_{prop}_lower']=np.nan; result[f'{role}_{prop}_upper']=np.nan
        if method_id is not None: result['method_id']=method_id''')
replace('case/models.py',"result.loc[valid_idx,'pass_robust_consensus']=robust&ad", "result.loc[valid_idx,'pass_robust_consensus']=robust&ad&gp&op")
replace('scripts/engine_entry.py',"        install_prior_metadata_bridge()", '''        from reinvent_plugins.components.comp_external_process import ExternalProcess
        ExternalProcess.no_cache=True
        install_prior_metadata_bridge()''')
replace('case/generation.py',"frame=pd.read_csv(folder/'sampled.csv')", "frame=pd.read_csv(folder/'sampled.csv',keep_default_na=False)")
replace('case/generation.py',"'oracle_loaded':False", "'oracle_loaded':False,'method_id':run['method_id'],'profile':run['profile']")
replace('case/generation.py',"checkpoint_hash=sha256(folder/'agent.chkpt')", "training_seconds=time.monotonic()-stage_start\n                checkpoint_hash=sha256(folder/'agent.chkpt')\n                sampling_start=time.monotonic()")
replace('case/generation.py',"sampled_rows=len(frame),sampler_keeps_invalid_and_duplicates=True))", '''sampled_rows=len(frame),sampler_keeps_invalid_and_duplicates=True,
                          training_seconds=training_seconds,sampling_seconds=time.monotonic()-sampling_start,
                          reward_query_count=sum(json.loads(line)['n_requested'] for line in (folder/'reward_ledger.jsonl').read_text().splitlines()),
                          reward_cache=False))''')
replace('case/metrics.py',"frame=pd.read_csv(input_path)", "frame=pd.read_csv(input_path,keep_default_na=False)")
replace('case/metrics.py',"if set(frame.method_id)-{'B0','M1'}: raise ValueError('Only B0 and M1 permitted')", '''if frame.empty or set(frame.method_id)-{'B0','M1'}: raise ValueError('Nonempty B0/M1 input required')
    numeric=pd.to_numeric(frame.seed,errors='coerce')
    if numeric.isna().any() or not np.isfinite(numeric).all() or (numeric<0).any() or (numeric!=np.floor(numeric)).any():
        raise ValueError('Finite nonnegative integer seeds required')
    frame['seed']=numeric.astype('int64')
    expected=set(map(tuple,frame[['method_id','seed']].drop_duplicates().to_numpy()))''')
replace('case/metrics.py',"if n_per_seed is not None:\n        if n_per_seed<=0:", "actual=set(map(tuple,frame[['method_id','seed']].drop_duplicates().to_numpy()))\n    if expected!=actual: raise ValueError('An entire method/seed group has no eligible structures')\n    if n_per_seed is not None:\n        if n_per_seed<=0:")
replace('case/metrics.py',"n_rows=len(frame),n_per_seed=n_per_seed,selection='canonicalize, remove invalid, first occurrence; no prediction used'", "n_rows=len(frame),n_per_seed=n_per_seed,groups=[list(k) for k in sorted(expected)],selection='canonical parent; first duplicate; fixed SHA256(seed:SMILES) subsample; no prediction used'")
replace('case/metrics.py',"numerical_comparison='changed target definitions/validation mean figures are not comparable to preceding report figures'", "numerical_comparison='Interpret using the target definitions and split manifests supplied with this run'")
p=ROOT/'case/data.py'
p.write_text(p.read_text()+'''\n\ndef verify_processed(root=ROOT):
    directory=Path(root)/'data/processed'
    manifest=json.loads((directory/'manifest.json').read_text())
    required={'dataset_group_A.csv','dataset_group_B.csv','dataset_combined.csv','measurements.csv.gz','rejected.csv'}
    if set(manifest['outputs'])!=required: raise ValueError('Unexpected processed manifest files')
    for name,expected in manifest['outputs'].items():
        if sha256(directory/name)!=expected: raise ValueError('Processed checksum mismatch: '+name)
    for name,expected in manifest['inputs'].items():
        if sha256(Path(root)/'data/raw'/name)!=expected: raise ValueError('Raw-to-processed input mismatch: '+name)
    frame=pd.read_csv(directory/'dataset_combined.csv')
    if len(frame)!=manifest['n_molecules'] or frame.canonical_smiles.duplicated().any(): raise ValueError('Processed row contract')
    if set(frame.source_group)!={'A','B'}: raise ValueError('Missing data group')
    a=set(frame.loc[frame.source_group=='A','identity_key']); b=set(frame.loc[frame.source_group=='B','identity_key'])
    if a&b: raise ValueError('Data groups overlap')
    for prop,spec in TARGETS.items():
        if frame.loc[frame.source_group!=spec['group'],prop].notna().any(): raise ValueError('Cross-group labels')
    return manifest
''')
replace('case/data.py',"outputs={p.name:sha256(p) for p in output.iterdir()", "outputs={p.name:sha256(p) for p in sorted(output.iterdir())")
write('case/integrity.py','''
"""Verify the published data, trained artifacts and experiment without unpickling."""
from pathlib import Path
import json
import pandas as pd
from .common import ROOT,TARGETS,SEEDS,sha256,save_json
from .data import verify_raw,verify_processed


def verify(root=ROOT,complete=False):
    root=Path(root); raw=verify_raw(root); processed=verify_processed(root)
    model=json.loads((root/'models/manifest.json').read_text())
    if model['targets']!=TARGETS or set(model['models'])!=set(TARGETS): raise ValueError('Model target contract')
    if model['source_data_sha256']!=sha256(root/'data/processed/dataset_combined.csv'): raise ValueError('Stale models')
    if model['splits_sha256']!=sha256(root/'models/splits.csv'): raise ValueError('Split hash')
    for prop,entry in model['models'].items():
        if sha256(root/'models'/entry['reference'])!=entry['reference_sha256']: raise ValueError('Reference hash: '+prop)
        for role in ['guidance','oracle']:
            if sha256(root/'models'/entry[role]['file'])!=entry[role]['sha256']: raise ValueError('Weights hash: '+prop)
    result=dict(raw_files=len(raw),raw_verified=True,processed_verified=True,models_verified=True,results_verified=False)
    marker=root/'results/experiment.json'
    if marker.exists():
        experiment=json.loads(marker.read_text())
        if experiment['status']!='completed' or experiment['smoke']: raise ValueError('Not a complete experiment')
        if experiment['model_manifest_sha256']!=sha256(root/'models/manifest.json'): raise ValueError('Stale experiment')
        for variant,subdir in [('none','main'),('no_AD','ablation_no_AD')]:
            directory=root/'results'/subdir
            provenance=json.loads((directory/'generated.provenance.json').read_text())
            if sha256(directory/'generated.csv')!=provenance['output_sha256']: raise ValueError('Generated hash')
            if provenance['model_manifest_sha256']!=sha256(root/'models/manifest.json'): raise ValueError('Stale predictions')
            if sha256(directory/'frozen_candidates.csv')!=provenance['frozen_sha256']: raise ValueError('Frozen input hash')
            f=pd.read_csv(directory/'generated.csv')
            methods=['B0','M1'] if variant=='none' else ['M1']
            expected={(m,s) for m in methods for s in SEEDS}
            if set(f.groupby(['method_id','seed']).groups)!=expected: raise ValueError('Incomplete seed/method groups')
            if not f.groupby(['method_id','seed']).size().eq(1000).all(): raise ValueError('Unequal evaluation counts')
            if f.duplicated(['method_id','seed','SMILES']).any(): raise ValueError('Duplicate final candidates')
            if not f['variant'].eq(variant).all(): raise ValueError('Wrong variant labels')
            for method,part in f.groupby('method_id'):
                if part.loc[part.novelty,'SMILES'].nunique()<1000: raise ValueError('Insufficient new structures')
            if (f.pass_constraints & ~f.pass_AD_SA).any(): raise ValueError('Out-of-domain successes')
            if (f.pass_robust_consensus & ~f.pass_constraints).any(): raise ValueError('Inconsistent robust status')
        if sha256(root/'generated.csv')!=sha256(root/'results/main/generated.csv'): raise ValueError('Root generated.csv differs')
        result['results_verified']=True
    elif complete: raise ValueError('No completed experiment')
    save_json(root/'results/verification.json',result)
    return result
''')
replace('pipeline.py',"sub.add_parser('verify'); sub.add_parser('build')", "ver=sub.add_parser('verify'); ver.add_argument('--complete',action='store_true'); sub.add_parser('build')")
replace('pipeline.py',"if args.command=='verify': verify_raw(restore_line_endings=False)", "if args.command=='verify':\n            from case.integrity import verify\n            print(verify(complete=args.complete))")
replace('pipeline.py',"        if args.command=='reproduce':\n            frame.to_csv(ROOT/'generated.csv',index=False)", "        # Reanalysis stays in results/reanalysis; never overwrite the controlled experiment.")
write('tests/test_release.py','''
import importlib.util
import json
import numpy as np
import pandas as pd
import pytest
from case.common import ROOT,TARGETS,utility_margin
from case.reward import score,front_reward
from case.metrics import freeze


def batch():
    f=pd.DataFrame({'valid':[True,True],'synthetic_accessibility':[2.,9.],
                    'dist_to_D_A':[.1,.1],'dist_to_D_B':[.1,.1]})
    for prop,spec in TARGETS.items():
        lo,hi=spec['low'],spec['high']
        f['pred_'+prop]=(lo+hi)/2 if lo is not None and hi is not None else (lo+1 if lo is not None else hi-1)
        f['unc_'+prop]=0.
    return f


def test_reward_does_not_mutate():
    f=batch(); before=f.copy(deep=True)
    for method in ['B0','M1']: assert score(f,method)[1]==0
    pd.testing.assert_frame_equal(f,before)
    f.index=[12,29]; assert score(f,'M1')[0]>0


def test_nan_reward_and_invalid_rank():
    f=batch(); f.loc[0,'pred_log_kp']=np.nan
    assert not score(f,'M1').any()
    with pytest.raises(ValueError): front_reward(np.nan,.5)
    with pytest.raises(ValueError): score(batch(),'B0',beta=np.inf)


def test_physical_bounds_are_not_competing_objectives():
    assert utility_margin([1.],TARGETS['pss_fraction'])[0]>utility_margin([.5],TARGETS['pss_fraction'])[0]
    for prop in ['skin_irritation','skin_sensitization']:
        assert utility_margin([0.],TARGETS[prop])[0]>utility_margin([.25],TARGETS[prop])[0]


def test_freeze_rejects_lost_groups(tmp_path):
    path=tmp_path/'in.csv'
    pd.DataFrame({'SMILES':['CCO','bad'],'method_id':['B0','M1'],'seed':[42,42]}).to_csv(path,index=False)
    with pytest.raises(ValueError,match='entire'): freeze(path,tmp_path/'out.csv',1)
    pd.DataFrame({'SMILES':['CCO'],'method_id':['B0'],'seed':['not_seed']}).to_csv(path,index=False)
    with pytest.raises(ValueError,match='seed'): freeze(path,tmp_path/'out.csv',1)


def test_engine_scoring_cache_disabled():
    text=(ROOT/'scripts/engine_entry.py').read_text()
    assert 'ExternalProcess.no_cache=True' in text


@pytest.mark.skipif(not (ROOT/'models/manifest.json').exists(),reason='Train first')
def test_all_invalid_schema():
    from case.models import Evaluator
    suite=Evaluator()
    frame=suite.evaluate(['bad',None],method_id='M1',seed=42)
    assert frame.status.eq('invalid').all()
    assert set('unc_'+p for p in TARGETS)<=set(frame)
    assert not frame.pass_constraints.any()
    assert suite.evaluate([]).empty


def test_processed_manifest_no_self_reference():
    manifest=json.loads((ROOT/'data/processed/manifest.json').read_text())
    assert 'manifest.json' not in manifest['outputs']
    from case.data import verify_processed
    verify_processed()
''')
print('Final source corrections applied')
