"""Exercise actual notebook data/recovery/export code with temporary fixtures, without detector training."""
import ast
import hashlib
import json
from pathlib import Path
import runpy
import shutil
from types import SimpleNamespace
import tempfile
from unittest.mock import patch
import zipfile

import torch

ROOT=Path(__file__).resolve().parents[1]
nb=json.loads((ROOT/'notebooks/object-detection.ipynb').read_text(encoding='utf-8'))
cells={c['metadata']['original_cell_index']:''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code'}
def definitions(index,names,scope):
    tree=ast.parse(cells[index])
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(nodes)==len(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),f'notebook-{index}','exec'),scope)
def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

with tempfile.TemporaryDirectory(prefix='recovery-check-',dir=ROOT/'data') as tmp:
    project=Path(tmp)/'project'
    project.mkdir()
    archive=project/'data/distribution/synthetic-scenes-v1.zip'
    reference=project/'assets/sources/dataset-manifest.json'
    checksum=reference.with_name('dataset-archive.sha256')
    dataset=project/'data/notebook-work/dataset'
    payload={'background/train/images/00000.jpg':b'image-fixture',
             'background/train/labels/00000.txt':b'0 .5 .5 .2 .3'}
    manifest={'seed':42,'files':{k:hashlib.sha256(v).hexdigest() for k,v in payload.items()}}
    fetched=[]
    def fetch(p):
        assert p==project
        fetched.append(p)
        archive.parent.mkdir(parents=True)
        reference.parent.mkdir(parents=True)
        text=json.dumps(manifest)
        reference.write_text(text)
        with zipfile.ZipFile(archive,'w') as zipped:
            for name,value in payload.items(): zipped.writestr(name,value)
            zipped.writestr('manifest.json',text)
        checksum.write_text(digest(archive))
    scope={'PROJECT_ROOT':project,'DATASET_DIR':dataset,'DATA_ARCHIVE':archive,
           'DATA_SHA_FILE':checksum,'DISTRIBUTION_DIR':archive.parent,'SEED':42,
           'IN_COLAB':False,'DRIVE_DIR':None,'json':json,'zipfile':zipfile,'shutil':shutil,
           'sha256_file':digest}
    definitions(20,{'verify_dataset','prepare_fixed_dataset'},scope)
    with patch.object(runpy,'run_path',return_value={'fetch_project':fetch}):
        dataset_id=scope['prepare_fixed_dataset']()
    assert len(fetched)==1 and dataset_id==digest(dataset/'manifest.json')
    print('PASS: fresh local setup downloads and verifies the archive; it does not regenerate scenes.')

    drive=Path(tmp)/'drive'
    metrics=drive/'metrics'; metrics.mkdir(parents=True)
    figures=drive/'figures'; figures.mkdir()
    models=drive/'models/original-notebook'; models.mkdir(parents=True)
    (metrics/'history.json').write_text('[]')
    (figures/'plot.png').write_bytes(b'figure-fixture')
    best=models/'best_model.pth'; best.write_bytes(b'best-fixture')
    last=models/'last_checkpoint.pth'; last.write_bytes(b'last-fixture')
    output=project/'outputs/original-notebook'
    yolo_project=drive/'yolo'
    scope.update(OUTPUT_DIR=output,METRIC_DIR=metrics,FIGURE_DIR=figures,
                 BEST_PATH=best,LAST_PATH=last,YOLO_PROJECT=yolo_project,DRIVE_DIR=drive)
    definitions(5,{'export_run_artifacts'},scope)
    result=scope['export_run_artifacts']()
    with zipfile.ZipFile(result) as zipped:
        assert {'models/best_model.pth','models/last_checkpoint.pth','metrics/history.json',
                'figures/plot.png','dataset-manifest.json'} <= set(zipped.namelist())
    assert digest(result)==digest(drive/'run-artifacts.zip')
    print('PASS: export works before YOLO and includes Drive-hosted metrics, figures and checkpoints.')

    yolo_run=yolo_project/'train'
    weights=yolo_run/'weights'; weights.mkdir(parents=True)
    yolo_last=weights/'last.pt'
    saved={'epoch':13,'train_args':{'epochs':100,'data':'/previous/runtime/data.yaml'},'model':torch.tensor([1.,2.])}
    torch.save(saved,yolo_last)
    before=digest(yolo_last)
    calls=[]
    class YoloFixture:
        def __init__(self,path): self.path=Path(path)
        def train(self,**kwargs):
            restored=torch.load(self.path,weights_only=False)
            assert torch.equal(restored['model'],saved['model'])
            assert restored['epoch']==13
            assert restored['train_args']['data']==str(project/'data.yaml')
            calls.append(kwargs)
    scope.update(torch=torch,YOLO=YoloFixture,YOLO_DATA=project/'data.yaml',DATASET_ID=dataset_id,
                 TRAIN_YOLO=True,NUM_WORKERS=0,USE_AMP=False,SEED=42,
                 device=SimpleNamespace(type='cpu'),train_loader=SimpleNamespace(batch_size=16))
    exec(compile(cells[49],'actual-yolo-resume-cell','exec'),scope)
    assert calls[0]['save_dir']==str(yolo_run) and calls[0]['workers']==0 and calls[0]['resume'] is True
    assert digest(yolo_last)==before
    print('PASS: relocated YOLO resume retains weights/epoch, uses current paths, and preserves the original checkpoint.')
    saved['epoch']=99
    torch.save(saved,yolo_last)
    (weights/'best.pt').write_bytes(b'best-fixture')
    calls.clear()
    exec(compile(cells[49],'completed-yolo-checkpoint','exec'),scope)
    assert not calls
    print('PASS: final-epoch checkpoint does not attempt an invalid extra resume.')
print('No full model training was executed.')
