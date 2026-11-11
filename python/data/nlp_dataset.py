"""NLP datasets for PyGo training."""
import os,json,random
from typing import List,Dict,Optional,Iterator
from dataclasses import dataclass

@dataclass
class TextSample:
    text:str; label:Optional[int]=None; meta:Optional[Dict]=None

class TextDataset:
    def __init__(self,texts,labels=None): self.texts=texts; self.labels=labels
    def __len__(self): return len(self.texts)
    def __getitem__(self,i): return TextSample(self.texts[i],self.labels[i] if self.labels else None)
    @classmethod
    def from_jsonl(cls,path,text_key="text",label_key=None):
        texts,labels=[],[]
        with open(path,encoding="utf-8") as f:
            for l in f:
                try:
                    o=json.loads(l); texts.append(o.get(text_key,""))
                    if label_key: labels.append(o.get(label_key))
                except: pass
        return cls(texts,labels or None)

class TokenizedDataset:
    def __init__(self,ids:List[List[int]],max_len=2048,stride=None,eos=2):
        self.max_len=max_len; self.stride=stride or max_len
        all_=[t for seq in ids for t in seq+[eos]]
        self.chunks=[all_[i:i+max_len] for i in range(0,len(all_)-max_len,self.stride)]
    def __len__(self): return len(self.chunks)
    def __getitem__(self,i):
        c=self.chunks[i]; return {"input_ids":c[:-1],"labels":c[1:],"attention_mask":[1]*(len(c)-1)}

class StreamingTextDataset:
    def __init__(self,paths,buf=10000,seed=42):
        self.paths=paths; self.buf=buf; self.rng=random.Random(seed)
    def __iter__(self) -> Iterator[TextSample]:
        buf=[]
        for path in self.paths:
            with open(path,encoding="utf-8",errors="replace") as f:
                for line in f:
                    line=line.rstrip()
                    if not line: continue
                    try: text=json.loads(line).get("text","")
                    except: text=line
                    buf.append(TextSample(text))
                    if len(buf)>=self.buf: self.rng.shuffle(buf); yield from buf; buf=[]
        if buf: self.rng.shuffle(buf); yield from buf
