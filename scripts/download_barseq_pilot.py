#!/usr/bin/env python3
"""Download the version-1 BARseq pilot and dictionary; verify publisher checksums."""
import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from paper.data_loaders.barseq_pilot import data_directory

FILES = [{'filename': 'variables.docx', 'url': 'https://data.mendeley.com/public-files/datasets/8bhhk7c5n9/files/22175819-b327-4e04-9f01-406914efec8c/file_downloaded', 'sha256': 'c908d82e2e2801207ece98d95dd4b5931c43e4155a2e955ab721c71117f6de76', 'size': 57803}, {'filename': 'wholecortexBARseq_filt_neurons_fixedbent_CCF.mat', 'url': 'https://data.mendeley.com/public-files/datasets/8bhhk7c5n9/files/2acb6643-d0b5-4270-82d9-701a3fac5dbd/file_downloaded', 'sha256': 'e54935dc1d25314c84c60be899b0781fe9160b5b8bb449a2877d6663184b8a0f', 'size': 195665608}]

def sha256(path):
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''): digest.update(chunk)
    return digest.hexdigest()

def download(target, cache_dir=None):
    target.mkdir(parents=True,exist_ok=True)
    for entry in FILES:
        path=target/entry['filename']
        if path.exists():
            if path.stat().st_size==entry['size'] and sha256(path)==entry['sha256']:
                print('Verified existing:',path); continue
            raise RuntimeError(f'Existing file failed checksum: {path}; move it aside before retrying')
        temporary=path.with_suffix(path.suffix+'.part')
        source=cache_dir/entry['filename'] if cache_dir else None
        if source and source.exists():
            if source.stat().st_size!=entry['size'] or sha256(source)!=entry['sha256']:
                raise RuntimeError(f'Cached source failed checksum: {source}')
            shutil.copyfile(source,temporary)
        else:
            print('Downloading:',entry['filename'],f"({entry['size']/1e6:.4g} MB)",flush=True)
            request=urllib.request.Request(entry['url'],headers={'User-Agent':'spatsplat BARseq downloader'})
            with urllib.request.urlopen(request,timeout=120) as response, temporary.open('wb') as out:
                shutil.copyfileobj(response,out,1024*1024)
        if temporary.stat().st_size!=entry['size'] or sha256(temporary)!=entry['sha256']:
            raise RuntimeError(f'Download failed checksum: {temporary}')
        temporary.replace(path)
    (target/'download_manifest.json').write_text(json.dumps(dict(
        dataset='https://data.mendeley.com/datasets/8bhhk7c5n9/1',version=1,files=FILES),indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir',type=Path,help='Optional directory of already downloaded files')
    args=parser.parse_args()
    download(data_directory(),args.cache_dir)
