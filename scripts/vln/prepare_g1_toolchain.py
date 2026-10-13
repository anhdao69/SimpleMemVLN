"""Install isolated, hash-verified CUDA 12.9 build components for CPU AdamW.

No driver, system symlink, shared Python package or existing toolkit is changed.
Matching cuBLAS/cuRAND already bundled with the pinned PyTorch are reused by
read-only symlinks; other versions are never silently substituted.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import urllib.request


def sha256(path):
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(8<<20),b''):digest.update(block)
    return digest.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args()
    root=Path(a.root).resolve();prefix=root/'cuda-12.9.1'
    if prefix.exists():raise FileExistsError(prefix)
    artifacts=root/'cuda-installers/g1-components-12.9.1';artifacts.mkdir(parents=True,exist_ok=True)
    base='https://developer.download.nvidia.com/compute/cuda/redist/'
    with urllib.request.urlopen(base+'redistrib_12.9.1.json',timeout=30) as response:
        metadata=json.load(response)
    (artifacts/'redistrib_12.9.1.json').write_text(json.dumps(metadata,indent=2))
    site=next((root/'SimpleMemVLN/.venv/lib').glob('python*/site-packages'))
    reused={};components=['cuda_nvcc','cuda_cudart','cuda_cccl']
    for component,package,library in [('libcublas','nvidia_cublas_cu12','cublas'),
                                      ('libcurand','nvidia_curand_cu12','curand')]:
        expected=metadata[component]['version']
        dist=site/f'{package}-{expected}.dist-info'
        if dist.exists():reused[component]=dict(version=expected,path=str(site/'nvidia'/library/'lib'))
        else:components.append(component)
    archives=[]
    for component in components:
        record=metadata[component]['linux-x86_64'];relative=record['relative_path']
        assert relative.startswith(component+'/') and '..' not in Path(relative).parts
        path=artifacts/Path(relative).name
        if not path.exists() or sha256(path)!=record['sha256']:
            print('DOWNLOAD',component,record['size'],flush=True)
            with urllib.request.urlopen(base+relative,timeout=60) as response,path.open('wb') as output:
                for block in iter(lambda:response.read(1<<20),b''):output.write(block)
        assert path.stat().st_size==int(record['size']) and sha256(path)==record['sha256'],component
        with tarfile.open(path) as archive:
            assert all(not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in archive)
        archives.append(path)
    prefix.mkdir()
    for path in archives:
        subprocess.run(['tar','-xJf',str(path),'--strip-components=1','-C',str(prefix)],check=True)
    libraries=prefix/'lib';libraries.mkdir(exist_ok=True)
    for record in reused.values():
        includes=Path(record['path']).parent/'include'
        if includes.exists():
            for header in includes.iterdir():
                target=prefix/'include'/header.name
                if not target.exists():target.symlink_to(header.resolve())
        for library in Path(record['path']).glob('*.so*'):
            target=libraries/library.name
            if not target.exists():target.symlink_to(library.resolve())
        # CUDA development linker names are absent from some runtime wheels.
        for stem in ('libcublas','libcublasLt','libcurand'):
            candidates=sorted(libraries.glob(stem+'.so.*'))
            if candidates and not (libraries/(stem+'.so')).exists():
                (libraries/(stem+'.so')).symlink_to(candidates[0].name)
    if not (prefix/'lib64').exists():(prefix/'lib64').symlink_to('lib')
    record=dict(cuda='12.9.1',components={c:metadata[c]['linux-x86_64'] for c in components},reused=reused)
    (prefix/'g1_build_manifest.json').write_text(json.dumps(record,indent=2)+'\n')
    subprocess.run([str(prefix/'bin/nvcc'),'--version'],check=True)
    print('ISOLATED_G1_CUDA_READY',prefix,flush=True)


if __name__=='__main__':main()
