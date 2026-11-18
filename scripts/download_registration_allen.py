#!/usr/bin/env python3
"""Populate the Allen cache needed by the self-contained registration notebook."""
import argparse
import json
import os
from pathlib import Path
from dotenv import load_dotenv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', help='Optional exact ABC manifest name from a prior run')
    args = parser.parse_args()
    load_dotenv(Path(__file__).resolve().parents[1]/'.env')
    base = os.getenv('SPATSPLAT_DATA_DIR')
    if not base or base == '/path/to/spatsplat/data':
        raise RuntimeError('Set an absolute SPATSPLAT_DATA_DIR in .env first')
    root = Path(os.getenv('SPATSPLAT_ABC_ATLAS_DIR', str(Path(base)/'raw/abc_atlas'))).expanduser()
    if not root.is_absolute():
        raise ValueError('Use absolute data/cache paths')
    from abc_atlas_access.abc_atlas_cache.abc_project_cache import AbcProjectCache
    cache = AbcProjectCache.from_cache_dir(root)
    if args.manifest:
        cache.load_manifest(args.manifest)
    directory = 'MERFISH-C57BL6J-638850'
    for dataset, filename in ((directory, 'cell_metadata'),
                              (directory+'-CCF', 'reconstructed_coordinates'),
                              ('Allen-CCF-2020', 'parcellation_to_parcellation_term_membership_acronym')):
        print('Fetching metadata:', dataset, filename, flush=True)
        cache.get_metadata_dataframe(directory=dataset, file_name=filename)
    print('Fetching released log2 expression (large file)', flush=True)
    path = cache.get_file_path(directory=directory, file_name='C57BL6J-638850/log2')
    record = dict(manifest=cache.current_manifest, directory=directory,
                  expression_file='C57BL6J-638850/log2', expression_path=str(path))
    (root/'registration_download.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
