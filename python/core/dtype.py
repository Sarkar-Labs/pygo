"""Data types for PyGo tensors."""
from enum import IntEnum

class DType(IntEnum):
    Float32=0; Float16=1; BFloat16=2; Float64=3; Int8=4; Int16=5
    Int32=6; Int64=7; UInt8=8; Bool=9; Complex64=10; Int4=12; NFloat4=13

    @property
    def itemsize(self):
        return {0:4,1:2,2:2,3:8,4:1,5:2,6:4,7:8,8:1,9:1,10:8,12:1,13:1}.get(int(self),4)

    @property
    def is_floating_point(self): return int(self) in (0,1,2,3)
    @property
    def is_integer(self): return int(self) in (4,5,6,7,8,12)
    def __repr__(self): return f"pygo.{self.name.lower()}"

float32=DType.Float32; float16=DType.Float16; bfloat16=DType.BFloat16
float64=DType.Float64; int8=DType.Int8; int32=DType.Int32; int64=DType.Int64
uint8=DType.UInt8; bool_=DType.Bool; int4=DType.Int4; nfloat4=DType.NFloat4

def promote_types(a,b):
    pri={0:0,8:1,4:2,5:3,6:4,7:5,1:6,2:7,3:8,9:9}
    if a.is_floating_point or b.is_floating_point:
        return a if pri.get(int(a),8)>=pri.get(int(b),8) else b
    return a if pri.get(int(a),4)>=pri.get(int(b),4) else b
