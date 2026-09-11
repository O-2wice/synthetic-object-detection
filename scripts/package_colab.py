"""Package the preserved notebook, saved assets and exact dataset for a private Colab run."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
dataset=ROOT/'data/distribution/synthetic-scenes-v1.zip'

def digest(path):
    with path.open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()

assert digest(dataset)==(ROOT/'assets/sources/dataset-archive.sha256').read_text().strip()
manifest=json.loads((ROOT/'assets/sources/dataset-manifest.json').read_text())
with zipfile.ZipFile(dataset) as archive:
    assert set(archive.namelist())==set(manifest['files'])|{'manifest.json'}
    for name,expected in manifest['files'].items():
        with archive.open(name) as stream:
            assert hashlib.file_digest(stream,'sha256').hexdigest()==expected,name
files=[ROOT/name for name in ['notebooks/object-detection.ipynb','requirements.txt','README.md','COLAB.md','DATA.md',
                             'scripts/verify_assets.py','scripts/check_original_notebook.py','scripts/package_colab.py',
                             'scripts/fetch_github_assets.py','scripts/check_notebook_privacy.py',
                             'scripts/check_notebook_recovery.py']]
files.extend(p for p in (ROOT/'assets').rglob('*') if p.is_file())
files.append(dataset)
destination=ROOT/'data/distribution/colab-project.zip'
with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_STORED) as archive:
    for file in sorted(files):
        archive.write(file,Path(ROOT.name)/file.relative_to(ROOT))
print(f'Verified all {len(manifest["files"])} dataset files inside the archive.')
print(f'Colab bundle: {destination} ({destination.stat().st_size/1024**2:.1f} MiB)')
