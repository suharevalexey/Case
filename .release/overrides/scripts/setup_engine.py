"""Install pinned REINVENT4 and verify the committed environment lock."""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import urllib.request
import venv
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from case.common import save_json,sha256
from case.generation import ENGINE_COMMIT,PRIOR_COMMIT,validate_prior


def download(url,path):
    temporary=path.with_suffix(path.suffix+'.part')
    with urllib.request.urlopen(url,timeout=120) as response,temporary.open('wb') as stream:
        shutil.copyfileobj(response,stream)
    temporary.replace(path)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--cuda-index',default=None)
    args=parser.parse_args()
    external=ROOT/'external'; external.mkdir(exist_ok=True)
    source=external/'REINVENT4'; archive=external/'reinvent-source.zip'
    if not source.exists():
        download(f'https://github.com/MolecularAI/REINVENT4/archive/{ENGINE_COMMIT}.zip',archive)
        with zipfile.ZipFile(archive) as z:
            for name in z.namelist():
                if Path(name).is_absolute() or '..' in Path(name).parts: raise ValueError('Unsafe source archive')
            z.extractall(external)
        (external/f'REINVENT4-{ENGINE_COMMIT}').rename(source)
    project=(source/'pyproject.toml').read_bytes()
    blob=hashlib.sha1(f'blob {len(project)}\0'.encode()+project).hexdigest()
    if blob!='8ae5e897344601de6b43df451b5106cd39a44db8': raise ValueError('Source metadata checksum mismatch')
    prior=external/'reinvent.prior'
    if not prior.exists():
        download(f'https://raw.githubusercontent.com/MolecularAI/REINVENT4/{PRIOR_COMMIT}/priors/reinvent.prior',prior)
    validate_prior(prior)
    lock=ROOT/'requirements-engine-lock.txt'
    if not lock.is_file(): raise FileNotFoundError('The committed requirements-engine-lock.txt is required')
    original_hash=sha256(lock)
    # An optional GPU installation retains all non-Torch pins; it is a different runtime.
    effective=lock
    if args.cuda_index:
        text=lock.read_text().replace('torch==2.12.0+cpu','torch==2.12.0').replace('torchvision==0.27.0+cpu','torchvision==0.27.0')
        effective=external/'engine-cuda-requirements.txt'; effective.write_text(text)
    directory=ROOT/'.venv-engine'
    if not directory.exists(): venv.EnvBuilder(with_pip=True).create(directory)
    python=directory/('Scripts/python.exe' if sys.platform=='win32' else 'bin/python')
    def pip(*arguments): subprocess.run([str(python),'-m','pip',*arguments],check=True)
    pip('install','pip==26.2.1')
    index=args.cuda_index or 'https://download.pytorch.org/whl/cpu'
    pip('install','-r',str(effective),'--extra-index-url',index)
    # Dependencies are already resolved and pinned, including scipy for plotting.
    pip('install','--no-deps','-e',str(source))
    pip('check')
    versions=json.loads(subprocess.check_output([str(python),'-m','pip','list','--format=json'],text=True))
    normalize=lambda name:re.sub(r'[-_.]+','-',name).lower()
    installed={normalize(item['name']):item['version'] for item in versions}
    for line in effective.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith('#'): continue
        name,expected=line.split('==',1); actual=installed.get(normalize(name))
        if args.cuda_index and normalize(name) in {'torch','torchvision'}:
            actual=actual.split('+')[0] if actual else actual
        if actual!=expected: raise ValueError(f'Engine dependency mismatch: {name}: {actual} != {expected}')
    subprocess.run([str(python),str(ROOT/'scripts/engine_entry.py'),'--self-test',str(prior)],check=True)
    if sha256(lock)!=original_hash: raise ValueError('Committed lock unexpectedly changed')
    text=subprocess.check_output([str(python),'-m','pip','freeze','--exclude-editable'],text=True)
    (ROOT/'results').mkdir(exist_ok=True)
    (ROOT/'results/requirements-engine-resolved.txt').write_text(text,encoding='utf-8')
    files={str(p.relative_to(source)):sha256(p) for p in source.rglob('*.py') if '.git' not in p.parts}
    save_json(external/'engine_manifest.json',dict(source_commit=ENGINE_COMMIT,prior_commit=PRIOR_COMMIT,
        prior_sha256=sha256(prior),pyproject_git_blob=blob,python=str(python),source_files=files,
        entry_sha256=sha256(ROOT/'scripts/engine_entry.py'),packages=versions,lock_sha256=original_hash,
        effective_lock_sha256=sha256(effective),runtime='cuda' if args.cuda_index else 'cpu'))
    print('Pinned engine installed and committed lock verified; source lock left unchanged')

if __name__=='__main__': main()
