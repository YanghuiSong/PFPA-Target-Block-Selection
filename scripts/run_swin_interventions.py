#!/usr/bin/env python3
"""108 independent one-block Swin-T PFPA interventions under the frozen protocol."""
from __future__ import annotations
import csv, hashlib, json, os, sys, time
from pathlib import Path
os.environ.setdefault('CUDA_VISIBLE_DEVICES', os.environ.get('PFPA_VISIBLE_GPU', '0'))
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

ROOT=Path(os.environ.get('PFPA_DATA_ROOT', Path.cwd())).expanduser().resolve()
OUT=Path(os.environ.get('PFPA_OUTPUT_ROOT', str(ROOT/'outputs/pfpa_supplement_v2'))).expanduser().resolve()
SWIN=OUT/'swin_t'
sys.path.insert(0,str(Path(__file__).resolve().parent))
import probe_transformer_block_plasticity as core
sys.path.insert(0,str(Path(__file__).parent))
from swin_t_model import SwinTUPerNet,get_transformer_blocks

DOMAINS={
 'sugar_beet':('manifests/cropandweed_sugar_beet_train_unlabeled.csv','manifests/cropandweed_sugar_beet_val.csv'),
 'maize':('manifests/cropandweed_maize_train_unlabeled.csv','manifests/cropandweed_maize_val.csv'),
 'weedsgalore':('manifests/weedsgalore_train_unlabeled.csv','manifests/weedsgalore_val.csv'),
}
SEEDS=[2027,3407,4096]
EPSILON=0.001
CALIBRATION_CAP=128
SOURCE_ANCHOR_CAP=128
BATCH=4
IGNORE=255
SOURCE_CKPT=Path(os.environ.get('PFPA_SWIN_SOURCE_CHECKPOINT', str(SWIN/'source_training/swin_t_phenobench_seed3407_best.pt'))).expanduser().resolve()


def sha256_file(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()


def id_hash(ids):
 return hashlib.sha256(''.join(str(x)+'\n' for x in ids).encode('utf-8')).hexdigest()


def gpu_guard():
 if not torch.cuda.is_available():
  raise SystemExit('CUDA is required for the exhaustive Swin-T intervention runner')
 device=torch.cuda.current_device()
 free_bytes,total_bytes=torch.cuda.mem_get_info(device)
 name=torch.cuda.get_device_name(device)
 print(json.dumps({'cuda_precheck':'PASS','device':name,'free_bytes':free_bytes,'total_bytes':total_bytes}),flush=True)


def load_source(device):
 model=SwinTUPerNet()
 raw=torch.load(SOURCE_CKPT,map_location='cpu')
 missing,unexpected=model.load_state_dict(raw['model_state_dict'],strict=True)
 if missing or unexpected: raise RuntimeError(f'source checkpoint mismatch: missing={missing} unexpected={unexpected}')
 return model.to(device).eval()


def block_params(block):
 return [(n,p) for n,p in block.named_parameters()]


def flatten_parameter_grads(block):
 chunks=[]
 for p in block.parameters():
  if p.grad is None: chunks.append(torch.zeros_like(p).reshape(-1))
  else: chunks.append(p.grad.detach().reshape(-1))
 return torch.cat(chunks).double()


def target_grads_all(model,blocks,loader,device):
 model.train(False); model.zero_grad(set_to_none=True)
 for p in model.parameters(): p.requires_grad_(True)
 loss_sum=0.0; batches=0
 for x in loader:
  x=x.to(device,non_blocking=True)
  xf=torch.flip(x,dims=[3])
  logits=model(x)
  logits_f=torch.flip(model(xf),dims=[3])
  ent=core.prediction_entropy(logits)
  js=core.symmetric_js(logits,logits_f)
  loss=ent+0.5*js
  loss.backward()
  loss_sum+=float(loss.detach()); batches+=1
 if batches==0: raise RuntimeError('empty target calibration loader')
 grads={}
 for item in blocks:
  module=item['module']
  vec=flatten_parameter_grads(module)/batches
  grads[item['block_key']]=vec
 return grads,loss_sum/batches


def source_grads_all(model,blocks,loader,device):
 model.train(False); model.zero_grad(set_to_none=True)
 for p in model.parameters(): p.requires_grad_(True)
 loss_sum=0.0; batches=0
 for x,y in loader:
  x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True)
  logits=model(x)
  per_pixel=F.cross_entropy(logits.float(),y,reduction='none',ignore_index=IGNORE)
  losses=[]
  for cls in range(3):
   mask=(y==cls)
   if mask.any(): losses.append(per_pixel[mask].mean())
  if not losses: continue
  loss=torch.stack(losses).mean()
  loss.backward(); loss_sum+=float(loss.detach()); batches+=1
 if batches==0: raise RuntimeError('source anchor loader had no labeled batch')
 grads={}
 for item in blocks:
  grads[item['block_key']]=flatten_parameter_grads(item['module'])/batches
 return grads,loss_sum/batches


def make_delta(block,target_grad,source_grad):
 gt=target_grad.double(); gs=source_grad.double()
 theta=torch.cat([p.detach().reshape(-1) for p in block.parameters()]).double()
 theta_norm=float(theta.norm().item()); grad_norm=float(gt.norm().item()); source_norm=float(gs.norm().item())
 dot=float(torch.dot(gt,gs).item()); cos_before=dot/max(grad_norm*source_norm,1e-30)
 projected=gt.clone()
 if dot<0:
  projected=gt-(dot/max(float(torch.dot(gs,gs).item()),1e-20))*gs
 proj_norm=float(projected.norm().item())
 cos_after=float(torch.dot(projected,gs).item())/max(proj_norm*source_norm,1e-30)
 removed=float((gt-projected).norm().item())/max(grad_norm,1e-30)
 if proj_norm<=1e-12: raise RuntimeError('projected target gradient is numerically zero')
 flat_delta=(-EPSILON*theta_norm/proj_norm*projected).float()
 delta={}; offset=0
 for name,param in block.named_parameters():
  n=param.numel(); delta[name]=flat_delta[offset:offset+n].reshape_as(param).to(dtype=param.dtype,device='cpu').contiguous(); offset+=n
 if offset!=flat_delta.numel(): raise RuntimeError('delta length mismatch')
 return delta,{'parameter_norm':theta_norm,'target_gradient_norm':grad_norm,'response_product':theta_norm*grad_norm,
   'source_gradient_norm':source_norm,'target_source_dot':dot,'gradient_cosine_before':cos_before,
   'gradient_cosine_after':cos_after,'conflict_removed_fraction':removed,
   'projected_gradient_norm':proj_norm,'relative_displacement':EPSILON}


def sample_ids_write(path,ids):
 with path.open('w',newline='',encoding='utf-8') as f:
  w=csv.writer(f); w.writerow(['sample_id']); w.writerows([[x] for x in ids])
 return id_hash(ids)


def metrics_from_cm(cm):
 cm=np.asarray(cm,dtype=np.int64); tp=np.diag(cm).astype(np.float64); fp=cm.sum(0)-tp; fn=cm.sum(1)-tp
 den=tp+fp+fn; iou=np.divide(tp,den,out=np.full(3,np.nan),where=den>0)
 precision=np.divide(tp,tp+fp,out=np.full(3,np.nan),where=(tp+fp)>0)
 recall=np.divide(tp,tp+fn,out=np.full(3,np.nan),where=(tp+fn)>0)
 f1=np.divide(2*tp,2*tp+fp+fn,out=np.full(3,np.nan),where=(2*tp+fp+fn)>0)
 return {'mIoU':float(np.nanmean(iou)),'background_IoU':float(iou[0]),'crop_IoU':float(iou[1]),
   'weed_IoU':float(iou[2]),'crop_precision':float(precision[1]),'crop_recall':float(recall[1]),
   'crop_F1':float(f1[1]),'weed_precision':float(precision[2]),'weed_recall':float(recall[2]),
   'weed_F1':float(f1[2]),'confusion_matrix':cm.tolist()}


def evaluate(model,loader,device):
 model.eval(); cm=np.zeros((3,3),dtype=np.int64)
 with torch.inference_mode():
  for batch in loader:
   x=batch[0].to(device,non_blocking=True); y=batch[1].cpu().numpy()
   pred=model(x).argmax(1).cpu().numpy()
   for p,t in zip(pred,y):
    valid=(t<3)&(p<3)
    hist=np.bincount(t[valid].astype(np.int64)*3+p[valid].astype(np.int64),minlength=9).reshape(3,3)
    cm+=hist
 return metrics_from_cm(cm)


def apply_delta(block,delta):
 params=dict(block.named_parameters())
 if set(params)!=set(delta): raise RuntimeError('delta parameter names do not match selected block')
 original={name:p.detach().clone() for name,p in params.items()}
 with torch.no_grad():
  for name,p in params.items(): p.add_(delta[name].to(device=p.device,dtype=p.dtype))
 return original


def restore_block(block,original):
 with torch.no_grad():
  for name,p in block.named_parameters(): p.copy_(original[name])


def main():
 gpu_guard()
 protocol_path=Path(os.environ.get('PFPA_PROTOCOL_FILE',str(Path(__file__).resolve().parent.parent/'protocol.json'))).expanduser().resolve()
 if not protocol_path.is_file(): raise SystemExit(f'missing protocol file: {protocol_path}')
 protocol_sha=sha256_file(protocol_path)
 if not SOURCE_CKPT.is_file(): raise SystemExit(f'missing Swin source checkpoint: {SOURCE_CKPT}')
 source_sha=sha256_file(SOURCE_CKPT)
 device=torch.device('cuda:0')
 records=[]; pending=[]
 # Stage 1: calculate all unlabeled scores and independent parameter deltas before opening target-val labels.
 model=load_source(device); blocks=get_transformer_blocks(model)
 for domain,(target_rel,val_rel) in DOMAINS.items():
  for seed in SEEDS:
   group_dir=SWIN/domain/f'seed_{seed}'
   group_dir.mkdir(parents=True,exist_ok=True)
   cap=min(CALIBRATION_CAP,len(pd.read_csv(ROOT/target_rel)))
   target_ds=core.TrainImages(str(ROOT/target_rel),512,cap,seed,masks=False)
   source_ds=core.TrainImages(str(ROOT/'manifests/phenobench_train.csv'),512,SOURCE_ANCHOR_CAP,seed,masks=True)
   target_ids=target_ds.df.sample_id.astype(str).tolist(); source_ids=source_ds.df.sample_id.astype(str).tolist()
   target_id_sha=sample_ids_write(group_dir/'target_calibration_sample_ids.csv',target_ids)
   source_id_sha=sample_ids_write(group_dir/'source_anchor_sample_ids.csv',source_ids)
   target_loader=DataLoader(target_ds,batch_size=BATCH,shuffle=False,num_workers=4,pin_memory=True)
   source_loader=DataLoader(source_ds,batch_size=BATCH,shuffle=False,num_workers=4,pin_memory=True)
   print(f'GRADIENTS_START {domain} seed={seed} target={len(target_ds)} source={len(source_ds)}',flush=True)
   target_grads,target_loss=target_grads_all(model,blocks,target_loader,device)
   source_grads,source_loss=source_grads_all(model,blocks,source_loader,device)
   score_rows=[]
   for item in blocks:
    key=item['block_key']; block=item['module']
    theta=float(torch.cat([p.detach().reshape(-1) for p in block.parameters()]).double().norm().item())
    gnorm=float(target_grads[key].norm().item()); product=theta*gnorm
    delta,diag=make_delta(block,target_grads[key],source_grads[key])
    run_dir=group_dir/item['block_id'].replace('/','_')
    if run_dir.exists() and any(run_dir.iterdir()): raise SystemExit(f'ABORT refusing to overwrite {run_dir}')
    run_dir.mkdir(parents=True,exist_ok=True)
    delta_path=run_dir/'parameter_delta.pt'
    torch.save({'block_id':item['block_id'],'block_key':key,'source_checkpoint_sha256':source_sha,
                'epsilon':EPSILON,'delta':delta},delta_path)
    config={'domain':domain,'seed':seed,'backbone':'Swin-T + UPerNet','block_id':item['block_id'],
      'block_key':key,'stage':item['stage'],'stage_block':item['stage_block'],
      'parameter_count':item['parameter_count'],'source_checkpoint_sha256':source_sha,
      'target_train_manifest_sha256':sha256_file(ROOT/target_rel),
      'target_calibration_ids_sha256':target_id_sha,'source_anchor_manifest_sha256':sha256_file(ROOT/'manifests/phenobench_train.csv'),
      'source_anchor_ids_sha256':source_id_sha,'target_objective':'entropy + 0.5*symmetric_JS(horizontal_flip)',
      'source_anchor_objective':'mean present-class cross-entropy; ignore=255',
      'conflict_projection':'remove target-gradient component along source gradient iff dot<0',
      'epsilon':EPSILON,'update_count':1,'optimizer':'none','candidate_scope':'one encoder transformer block only',
      'target_labels_used_for_scoring_or_update':False,'delta_sha256':sha256_file(delta_path),**diag}
    (run_dir/'config.json').write_text(json.dumps(config,indent=2)+'\n',encoding='utf-8')
    score_row={'domain':domain,'seed':seed,'block':item['block_id'],'block_key':key,'stage':item['stage'],
      'stage_block':item['stage_block'],'parameter_count':item['parameter_count'],'P_theta':theta,'P_g':gnorm,'P_l':product,
      'target_loss':target_loss,'source_loss':source_loss,'target_calibration_images':len(target_ds),
      'source_anchor_images':len(source_ds),'target_calibration_ids_sha256':target_id_sha,
      'source_anchor_ids_sha256':source_id_sha,'source_checkpoint_sha256':source_sha,
      'delta_path':str(delta_path),'delta_sha256':sha256_file(delta_path),'run_dir':str(run_dir),
      'target_labels_used_for_score_or_update':False,'status':'DELTA_READY'}
    score_rows.append(score_row); pending.append((domain,seed,item,run_dir,delta_path,score_row))
   pd.DataFrame(score_rows).to_csv(group_dir/'gradient_scores.csv',index=False,encoding='utf-8-sig',float_format='%.10g')
   print(f'GRADIENTS_DONE {domain} seed={seed} target_loss={target_loss:.8g} source_loss={source_loss:.8g}',flush=True)
   del target_loader,source_loader,target_ds,source_ds,target_grads,source_grads
   torch.cuda.empty_cache()
 del model; torch.cuda.empty_cache()
 if len(pending)!=108: raise RuntimeError(f'expected 108 deltas, got {len(pending)}')
 delta_freeze={'status':'ALL_108_DELTAS_FROZEN_BEFORE_TARGET_VALIDATION_READ',
  'source_checkpoint_sha256':source_sha,'protocol_sha256':protocol_sha,
  'target_labels_used_for_scoring_or_updates':False,'n_deltas':len(pending),'deltas':[]}
 for domain,seed,item,run_dir,delta_path,score_row in pending:
  delta_freeze['deltas'].append({'domain':domain,'seed':seed,'block':item['block_id'],
   'delta_path':str(delta_path),'delta_sha256':sha256_file(delta_path),
   'config_sha256':sha256_file(run_dir/'config.json'),
   'target_calibration_ids_sha256':score_row['target_calibration_ids_sha256']})
 freeze_path=SWIN/'all_108_deltas_frozen_before_target_validation.json'
 if freeze_path.exists(): raise SystemExit(f'ABORT refusing to overwrite {freeze_path}')
 freeze_path.write_text(json.dumps(delta_freeze,indent=2)+'\n',encoding='utf-8')
 print('ALL_108_DELTAS_FROZEN_BEFORE_TARGET_VALIDATION_READ',flush=True)
 # Stage 2: only after every score/delta is frozen, read target validation masks for descriptive utility.
 model=load_source(device); blocks=get_transformer_blocks(model); by_key={x['block_key']:x for x in blocks}
 baseline={}; val_loaders={}
 for domain,(_,val_rel) in DOMAINS.items():
  val_ds=core.TrainImages(str(ROOT/val_rel),512,0,3407,masks=True)
  val_loaders[domain]=DataLoader(val_ds,batch_size=BATCH,shuffle=False,num_workers=4,pin_memory=True)
  before=evaluate(model,val_loaders[domain],device); baseline[domain]=before
  base_path=SWIN/'source_only'/domain/'target_validation_metrics.json'; base_path.parent.mkdir(parents=True,exist_ok=True)
  base_path.write_text(json.dumps({'domain':domain,'split':'target_validation_historically_viewed_posthoc',
     'source_checkpoint_sha256':source_sha,'metrics':before},indent=2)+'\n',encoding='utf-8')
  print(f'SOURCE_ONLY_VAL {domain} mIoU={before["mIoU"]:.8f}',flush=True)
 for domain,seed,item,run_dir,delta_path,score_row in pending:
  block=by_key[item['block_key']]['module']
  delta_obj=torch.load(delta_path,map_location='cpu')
  original=apply_delta(block,delta_obj['delta'])
  after=evaluate(model,val_loaders[domain],device)
  restore_block(block,original)
  before=baseline[domain]
  record={**score_row,'target_validation_before_mIoU':before['mIoU'],
    'target_validation_after_mIoU':after['mIoU'],
    'target_validation_delta_pp':100*(after['mIoU']-before['mIoU']),
    'target_validation_before_crop_IoU':before['crop_IoU'],'target_validation_after_crop_IoU':after['crop_IoU'],
    'target_validation_delta_crop_IoU_pp':100*(after['crop_IoU']-before['crop_IoU']),
    'target_validation_before_weed_IoU':before['weed_IoU'],'target_validation_after_weed_IoU':after['weed_IoU'],
    'target_validation_delta_weed_IoU_pp':100*(after['weed_IoU']-before['weed_IoU']),
    'target_validation_after_weed_recall':after['weed_recall'],
    'target_validation_confusion_matrix_json':json.dumps(after['confusion_matrix']),
    'target_validation_status':'POSTHOC_TARGET_VALIDATION_DESCRIPTIVE','status':'PASS'}
  (run_dir/'target_validation_metrics.json').write_text(json.dumps({'domain':domain,'seed':seed,'block':item['block_id'],
    'split':'target_validation_historically_viewed_posthoc','source_checkpoint_sha256':source_sha,
    'metrics':after},indent=2)+'\n',encoding='utf-8')
  records.append(record)
  print(f'VAL_DONE {domain} seed={seed} {item["block_id"]} gain_pp={record["target_validation_delta_pp"]:.5f}',flush=True)
  del delta_obj
 if len(records)!=108: raise RuntimeError(f'expected 108 evaluated block interventions, got {len(records)}')
 pd.DataFrame(records).to_csv(OUT/'swin_t_block_results.csv',index=False,encoding='utf-8-sig',float_format='%.10g')
 summary={'status':'PASS','n_interventions':len(records),'domains':list(DOMAINS),'seeds':SEEDS,
  'blocks_per_seed':12,'source_checkpoint_sha256':source_sha,
  'evidence_status':'POSTHOC_TARGET_VALIDATION_DESCRIPTIVE; target labels were never used for scoring or updates'}
 (SWIN/'scan_summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
 print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__': main()
