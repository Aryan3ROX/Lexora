import json
from pathlib import Path

_DIR = Path(__file__).parent / 'bpe_model'
_MERGES_FILE = _DIR / 'merges.json'
_VOCAB_FILE = _DIR / 'vocab.json'

with _MERGES_FILE.open('r', encoding='utf-8') as f:
    _merges_data = json.load(f)
_merges_list = _merges_data.get('merges', [])
merges = [ (tuple(item['pair']), item['id']) for item in _merges_list ]

with _VOCAB_FILE.open('r', encoding='utf-8') as f:
    _vocab_raw = json.load(f)
vocab = { int(k): bytes.fromhex(v) for k, v in _vocab_raw.items() }

def encode(text: str):
    ids = list(text.encode('utf-8'))
    for (a,b), nid in merges:
        out = []
        i = 0
        while i < len(ids):
            if i < len(ids)-1 and ids[i] == a and ids[i+1] == b:
                out.append(nid)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        ids = out
    return ids

def decode(token_ids):
    return b''.join(vocab[t] for t in token_ids).decode('utf-8', errors='replace')

__all__ = ['merges', 'vocab', 'encode', 'decode']