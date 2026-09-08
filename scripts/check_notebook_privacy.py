"""Check every notebook JSON string, including outputs and metadata, without printing potential secrets."""
import argparse
import ast
import json
from pathlib import Path
import re

from IPython.core.inputtransformer2 import TransformerManager
import nbformat

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('notebook',nargs='?',type=Path,default=ROOT/'notebooks/object-detection.ipynb')
args=parser.parse_args()
raw=args.notebook.read_text(encoding='utf-8-sig')
notebook=json.loads(raw)
patterns={
    'GitHub credential':r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,})',
    'API credential':r'\b(?:sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{20,}|AIza[0-9A-Za-z_-]{30,}|xox[baprs]-[0-9A-Za-z-]{15,})',
    'AWS access key':r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b',
    'private key':r'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----',
    'JWT':r'\beyJ[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{12,}',
    'URL credentials':r'https?://[^\s/@:]+:[^\s/@]+@',
    'credential assignment':r'''(?i)\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|password|client[_-]?secret|wandb_api_key)\s*[=:]\s*["'][^"'\s]{8,}["']''',
    'bearer credential':r'(?i)\bbearer\s+[A-Za-z0-9_.-]{20,}',
}
findings=[]
def walk(value,location='$'):
    if isinstance(value,dict):
        for key,child in value.items():
            if re.fullmatch(r'(?i)(?:api[_-]?key|access[_-]?token|auth[_-]?token|password|client[_-]?secret|wandb_api_key)',key) and isinstance(child,str) and len(child)>=8:
                findings.append((f'{location}.{key}','credential field'))
            walk(child,f'{location}.{key}')
    elif isinstance(value,list):
        for index,child in enumerate(value): walk(child,f'{location}[{index}]')
    elif isinstance(value,str):
        for name,pattern in patterns.items():
            if re.search(pattern,value): findings.append((location,name))
walk(notebook)
if findings:
    for location,name in findings: print(f'Potential {name} at {location}; value withheld.')
    raise SystemExit('Review potential secrets before publishing.')
nbformat.validate(nbformat.from_dict(notebook))
transformer=TransformerManager()
for cell in notebook['cells']:
    if cell['cell_type']=='code':
        source=''.join(cell['source'])
        compile(transformer.transform_cell(source),'notebook-cell','exec')
        assert not re.search(r'\b(?:import wandb|from wandb|wandb\.|log_to_wandb|use_wandb)\b',source)
assert '\u2014' not in raw,'Em dash still present'
print('PASS: no credential-pattern matches in cells, outputs or metadata; no tracking integration; all code compiles; no em dashes.')
print('Pattern scanning cannot rule out every possible secret; review unexpected outputs before publishing future runs.')
