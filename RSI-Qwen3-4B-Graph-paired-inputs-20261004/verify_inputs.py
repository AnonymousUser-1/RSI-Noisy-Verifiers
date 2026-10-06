"""Verify a copied paired-input bundle without GPU work or mutations."""
from pathlib import Path, PurePosixPath
import argparse, hashlib, json, os, sys

def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def verify(root, repo=None):
    root=Path(root).resolve();manifest=read(root/'PACKAGE_MANIFEST.json');protocol=read(root/'PROTOCOL.json')
    for item in manifest['files']:
        rel=PurePosixPath(item['path'])
        if rel.is_absolute() or '..' in rel.parts:raise ValueError('Invalid manifest path')
        path=root/rel
        if not path.resolve().is_relative_to(root) or path.is_symlink():raise ValueError('Unexpected path/link')
        if path.stat().st_size!=item['bytes'] or sha(path)!=item['sha256']:raise ValueError('File mismatch: '+item['path'])
    dataset=read(root/'graph/manifest.json')
    if dataset['task']!='graph' or dataset['seed']!=2027:raise ValueError('Wrong dataset')
    for name,expected in dataset['files'].items():
        path=root/'graph'/name
        if sha(path)!=expected:raise ValueError('Dataset changed: '+name)
        with path.open(encoding='utf8') as stream:count=sum(1 for line in stream if line.strip())
        if count!=dataset['counts'][Path(name).stem]:raise ValueError('Dataset count mismatch')
    pool_results=[]
    for i in range(5):
        block='b%02d'%i;path=root/'pools'/(block+'.jsonl');meta=read(str(path)+'.meta.json')
        if sha(path)!=meta['hash'] or meta['model']!=protocol['model'] or meta['revision']!=protocol['revision']:
            raise ValueError('Pool identity/hash mismatch')
        if meta['seed']['cli']!=i or meta['generation'].get('repetition_stop') is not None:
            raise ValueError('Unexpected first-pool generation identity')
        tasks=set();ids=set();truncated=0;count=0
        with path.open(encoding='utf8') as stream:
            for line in stream:
                r=json.loads(line);count+=1;tasks.add(r['task_id']);ids.add(r['id'])
                if type(r.get('truncated')) is not bool:raise ValueError('Missing truncated flag')
                truncated+=int(r['truncated'])
        if count!=16384 or len(tasks)!=2048 or len(ids)!=count:raise ValueError('Pool coverage mismatch')
        pool_results.append({'block':block,'answers':count,'prompts':len(tasks),'truncated':truncated})
    adapter=read(root/'shared_adapter/shared_adapter.json')
    if adapter['parameter_hash']!=protocol['adapter_parameter_hash'] or adapter['base_revision']!=protocol['revision']:
        raise ValueError('Common initialization differs')
    matched=read(root/'matching/matched_subsets.json')
    if matched['K']!=64:raise ValueError('Unexpected K')
    for block in ('b00','b01','b02','b03','b04'):
        for arm in ('R','S'):
            if len(matched['per_block'][block][arm])!=64:raise ValueError('Matching selection length differs')
    for version in ('legacy','new'):
        folder=root/('reference_out_audit_'+version);record=read(folder/'h.json')
        if sha(folder/'h.pt')!=record['h']['sha256']:raise ValueError('Reference gradient hash mismatch')
    bindings=None
    if repo:
        repo=Path(repo).resolve();sys.path.insert(0,str(repo))
        os.environ['RSI_BASE_PIN']=str(repo/'matched-dynamics/base_pin_qwen3-4b.json')
        from rsi.common import source_hash,load_config,require_explicit,verify_dataset
        from rsi.experiment import pin_config
        from run_matched_experiment import check_hf_inputs
        if source_hash()!=protocol['execution_source_hash']:raise ValueError('Wrong repository source')
        data=verify_dataset(root/'graph');bindings={}
        for version,name in [('legacy','legacy_audit_2048x8.json'),('new','audit_2048x8.json')]:
            path=root/'configs'/name;require_explicit(path,'iterative')
            cfg=pin_config(load_config(path,0,'hf'))
            bindings[version]=check_hf_inputs(cfg,data,root/'shared_adapter',root/('reference_out_audit_'+version))
    return {'status':'verified','files':len(manifest['files']),'pools':pool_results,
            'first_pools_use_new_repetition_rule':False,'reference_config_bindings_verified':bindings is not None}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default=str(Path(__file__).resolve().parent));p.add_argument('--repo')
    a=p.parse_args();print(json.dumps(verify(a.root,a.repo),ensure_ascii=False,indent=2))
