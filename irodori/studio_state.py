"""Studio extension: persistent Speaker Inversion training state, single GPU only."""
import hashlib
import json
import os
import random
from dataclasses import asdict
from pathlib import Path

import torch

VERSION = 1


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while data := f.read(8*1024*1024):
            h.update(data)
    return h.hexdigest()


def rng_state():
    return {'python': random.getstate(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(value):
    random.setstate(value['python'])
    torch.set_rng_state(value['torch'])
    if value['cuda']:
        if len(value['cuda']) != torch.cuda.device_count():
            raise ValueError('再開時のCUDAデバイス構成が異なります。')
        torch.cuda.set_rng_state_all(value['cuda'])


def save(directory, model, optimizer, scheduler, step, model_cfg, train_cfg, loader, runtime):
    """Atomically commit small mutable state; frozen base weights remain external."""
    directory=Path(directory)
    context=json.loads(os.environ['STUDIO_CONTEXT'])
    payload={'version':VERSION,'step':step,
             'trainable':{n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad},
             'buffers':{n:b.detach().cpu().clone() for n,b in model.named_buffers()},
             'optimizer':optimizer.state_dict(), 'scheduler':None if scheduler is None else scheduler.state_dict(),
             'model_config':asdict(model_cfg),'train_config':asdict(train_cfg),
             'dataloader':loader,'runtime':runtime,'rng':rng_state(),'context':context,
             'torch_version':str(torch.__version__)}
    temporary=directory/'resume_state.tmp'
    with temporary.open('wb') as f:
        torch.save(payload,f)
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(directory/'resume_state.pt')
    info={'step':step,'target_steps':train_cfg.max_steps,'state_file':str(directory/'resume_state.pt')}
    temporary=directory/'resume_info.tmp'
    temporary.write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    temporary.replace(directory/'resume_info.json')
    print(f'[Studio] 再開用状態を保存: step={step} / {directory / "resume_state.pt"}',flush=True)


def restore(path,model,optimizer,scheduler,model_cfg,train_cfg):
    state=torch.load(path,map_location='cpu',weights_only=True)
    if state['version'] != VERSION or state['torch_version'] != str(torch.__version__):
        raise ValueError('再開状態の形式またはPyTorchバージョンが異なります。')
    if state['model_config'] != asdict(model_cfg) or state['train_config'] != asdict(train_cfg):
        raise ValueError('完全再開時は学習設定を変更できません。')
    if state['context'] != json.loads(os.environ['STUDIO_CONTEXT']):
        raise ValueError('基盤モデル・学習素材またはコードが変更されています。完全再開できません。')
    parameters=dict(model.named_parameters())
    expected={n for n,p in parameters.items() if p.requires_grad}
    if expected != set(state['trainable']):
        raise ValueError('学習対象パラメーターが異なります。')
    with torch.no_grad():
        for name,value in state['trainable'].items():
            parameters[name].copy_(value)
        buffers=dict(model.named_buffers())
        for name,value in state['buffers'].items():
            if name not in buffers or buffers[name].shape != value.shape:
                raise ValueError('モデルバッファーの構成が異なります。')
            buffers[name].copy_(value)
    optimizer.load_state_dict(state['optimizer'])
    if scheduler is not None:
        scheduler.load_state_dict(state['scheduler'])
    return state


def restored_iterator(loader,state):
    iterator=iter(loader)
    generator=state['dataloader'].get('studio_generator')
    if generator is not None:
        loader.generator.set_state(generator)
    restore_rng(state['rng'])
    return iterator
