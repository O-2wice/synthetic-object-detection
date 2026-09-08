"""Verify bundled character PNGs and download/decode the notebook's remote illustrations."""
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile
from concurrent.futures import ThreadPoolExecutor

from PIL import Image
import requests

ROOT=Path(__file__).resolve().parents[1]
notebook=json.loads((ROOT/'notebooks/object-detection.ipynb').read_text(encoding='utf-8'))
bundle=ROOT/'data/distribution/colab-project.zip'
with zipfile.ZipFile(bundle) as archive:
    for name in ['Waldo.png','Wenda.png','Wizard Whitebeard.png']:
        local=(ROOT/'assets/objects'/name).read_bytes()
        packed=archive.read(f'{ROOT.name}/assets/objects/{name}')
        assert local==packed,name
        with Image.open(io.BytesIO(packed)) as im:
            im.load()
            assert im.mode=='RGBA' and im.getchannel('A').getbbox(),name
            print(f'Bundled {name}: decoded {im.size}, valid alpha, identical to local PNG',flush=True)

urls=[]
for cell in notebook['cells']:
    if cell['cell_type']=='markdown':
        urls.extend(re.findall(r'<img[^>]+src="(https?://[^"]+)"',''.join(cell['source'])))
destination=ROOT/'assets/illustrations'
destination.mkdir(parents=True,exist_ok=True)

def check(item):
    index,url=item
    result={'url':url}
    try:
        response=requests.get(url,timeout=30)
        result['status']=response.status_code
        response.raise_for_status()
        with Image.open(io.BytesIO(response.content)) as im:
            im.load()
            result.update(format=im.format,size=list(im.size))
            suffix={'PNG':'.png','JPEG':'.jpg','WEBP':'.webp'}[im.format]
        file=destination/f'original-illustration-{index+1}{suffix}'
        file.write_bytes(response.content)
        result.update(saved=file.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(response.content).hexdigest(),valid=True)
    except Exception as exc:
        result.update(valid=False,error=str(exc))
    return result

with ThreadPoolExecutor(max_workers=3) as pool:
    results=list(pool.map(check,enumerate(urls)))
for result in results: print(json.dumps(result),flush=True)
(ROOT/'assets/sources/notebook-illustrations.json').write_text(json.dumps(results,indent=2)+'\n',encoding='utf-8')
if not all(r['valid'] for r in results): raise SystemExit('At least one remote illustration failed verification.')
