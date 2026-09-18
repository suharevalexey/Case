"""REINVENT entry with stable semantic checkpoint hashes and untouched upstream files.

The digest covers network tensors, architecture, vocabulary and metadata.
The caller also verifies SHA-256 of the COMPLETE checkpoint file.
"""
from __future__ import annotations
import copy
import hashlib
import json
import sys
import time
import torch
from reinvent.models import meta_data as md
import reinvent.models as models

FORMAT='case.sha256.network_vocab_meta.v1'
ORIGINAL_CHECK=md.check_valid_hash


def semantic_hash(data):
    h=hashlib.sha256()
    meta=data.get('metadata')
    meta=meta.as_dict() if hasattr(meta,'as_dict') else copy.deepcopy(meta or {})
    meta.pop('hash_id',None); meta.pop('hash_id_format',None)
    header={k:data.get(k) for k in ['model_type','version','network_params','max_sequence_length','vocabulary']}
    vocabulary=data.get('vocabulary')
    if hasattr(vocabulary,'get_dictionary'):
        header['vocabulary']=vocabulary.get_dictionary()
    tokenizer=data.get('tokenizer')
    header['tokenizer_type']=type(tokenizer).__module__+'.'+type(tokenizer).__qualname__
    header['metadata']=meta
    h.update(json.dumps(header,sort_keys=True,separators=(',',':'),ensure_ascii=False,default=str).encode())
    network=md._get_network(data)
    for key in sorted(network):
        tensor=network[key].detach().cpu().contiguous()
        h.update(json.dumps([key,str(tensor.dtype),list(tensor.shape)],separators=(',',':')).encode())
        h.update(tensor.numpy().tobytes())
    return h.hexdigest()


def update(data,comment='',write_update=True):
    result=dict(data)
    metadata=data['metadata']
    metadata=metadata.as_dict() if hasattr(metadata,'as_dict') else copy.deepcopy(metadata)
    if write_update: metadata.setdefault('updates',[]).append(time.time())
    if comment: metadata.setdefault('comments',[]).append(comment)
    metadata['hash_id']=None; metadata['hash_id_format']=FORMAT
    result['metadata']=metadata; metadata['hash_id']=semantic_hash(result)
    return result


def check(data):
    metadata=data.get('metadata')
    metadata=metadata.as_dict() if hasattr(metadata,'as_dict') else metadata
    if metadata and metadata.get('hash_id_format')==FORMAT:
        return metadata['hash_id']==semantic_hash(data)
    return ORIGINAL_CHECK(data)


md.update_model_data=update
md.check_valid_hash=check
models.update_model_data=update
models.check_valid_hash=check

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--self-test':
        import tempfile
        from pathlib import Path
        data=torch.load(sys.argv[2],map_location='cpu',weights_only=False)
        if not data.get('metadata'):
            data['metadata']={'hash_id':None,'hash_id_format':'','model_id':'test','origina_data_source':'verified prior','creation_date':0.,'updates':[],'comments':[]}
        sealed=update(data,comment='round-trip self-test',write_update=False)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'state.chkpt'; torch.save(sealed,path)
            loaded=torch.load(path,map_location='cpu',weights_only=False)
            assert check(loaded), 'Round-trip changed semantic hash'
            first=next(iter(md._get_network(loaded).values()))
            first.view(-1)[0]+=1
            assert not check(loaded), 'Weight mutation was not detected'
        print('Checkpoint round-trip and tamper detection passed')
    else:
        from reinvent.Reinvent import main_script
        main_script()
