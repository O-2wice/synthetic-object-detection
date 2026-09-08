"""Download the saved source images and exact dataset from public GitHub."""
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import zipfile
import requests

REPOSITORY = 'O-2wice/synthetic-object-detection'
DATASET_TAG = 'dataset-v1'
DATASET_NAME = 'synthetic-scenes-v1.zip'

def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()

def fetch_project(project, revision='main'):
    project = Path(project)
    project.mkdir(parents=True,exist_ok=True)
    print('Downloading saved project assets from GitHub...',flush=True)
    url = f'https://github.com/{REPOSITORY}/archive/{revision}.zip'
    with requests.get(url,timeout=(20,120)) as response:
        if response.status_code != 200:
            raise RuntimeError(f'GitHub repository download failed (HTTP {response.status_code}).')
        contents = response.content
    with zipfile.ZipFile(io.BytesIO(contents)) as archive:
        for member in archive.infolist():
            parts = PurePosixPath(member.filename).parts
            if len(parts)<2 or member.is_dir(): continue
            relative = PurePosixPath(*parts[1:])
            if relative.parts[0] not in {'assets','scripts'} and str(relative)!='requirements.txt':
                continue
            target = (project/str(relative)).resolve()
            if not target.is_relative_to(project.resolve()):
                raise ValueError('Invalid repository archive path')
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(archive.read(member))
    manifest = json.loads((project/'assets/sources/dataset-manifest.json').read_text())
    for relative,expected in manifest['source_hashes'].items():
        if sha256(project/relative)!=expected:
            raise ValueError(f'Source checksum mismatch: {relative}')
    destination = project/'data/distribution'/DATASET_NAME
    expected = (project/'assets/sources/dataset-archive.sha256').read_text().strip()
    if destination.exists():
        if sha256(destination)!=expected:
            raise ValueError('Existing dataset archive differs from the frozen dataset; inspect it before replacing it.')
        print('Existing dataset archive verified.',flush=True)
        return destination
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.with_suffix('.download')
    url=f'https://github.com/{REPOSITORY}/releases/download/{DATASET_TAG}/{DATASET_NAME}'
    print('Downloading the exact dataset (about 578 MiB)...',flush=True)
    downloaded=0
    next_notice=50*1024**2
    with requests.get(url,stream=True,timeout=(20,120)) as response:
        if response.status_code!=200:
            raise RuntimeError(f'GitHub dataset download failed (HTTP {response.status_code}).')
        with temporary.open('wb') as stream:
            for chunk in response.iter_content(1024**2):
                if not chunk: continue
                stream.write(chunk)
                downloaded+=len(chunk)
                if downloaded>=next_notice:
                    print(f'  {downloaded/1024**2:.0f} MiB downloaded',flush=True)
                    next_notice+=50*1024**2
    if sha256(temporary)!=expected:
        raise ValueError('Downloaded dataset checksum mismatch; the incomplete file was not installed.')
    temporary.replace(destination)
    print('Character PNGs, backgrounds and exact dataset verified.',flush=True)
    return destination
