package tensor

import "fmt"

// QuantScheme specifies the quantisation granularity
type QuantScheme int

const (
	PerTensor QuantScheme = iota
	PerChannel
	PerGroup
)

// QuantizedTensor holds int8/int4 weights + scale/zero-point
type QuantizedTensor struct {
	data      *Tensor
	scale     *Tensor
	zeroPoint *Tensor
	bits      int
	scheme    QuantScheme
	groupSize int
	origShape Shape
	origDtype DType
}

func QuantToInt8(t *Tensor) *QuantizedTensor {
	absMax := t.Norm(1, -1, false).Item()
	scale := absMax / 127.0
	qt := &QuantizedTensor{bits: 8, scheme: PerTensor, origShape: t.Shape(), origDtype: t.DType()}
	if scale > 0 {
		qt.data = t.MulScalar(1.0/scale).Clamp(-127, 127).CastTo(Int8)
	} else {
		qt.data = Zeros(t.Shape(), WithDType(Int8), WithDevice(t.Device()))
	}
	qt.scale = FromSlice([]float32{float32(scale)}, Shape{1})
	return qt
}

func (q *QuantizedTensor) Dequantize() *Tensor {
	return q.data.CastTo(Float32).Mul(q.scale)
}
func (q *QuantizedTensor) CompressionRatio() float64 {
	return float64(q.origDtype.ItemSize()) / float64(q.bits) * 8
}
func (q *QuantizedTensor) String() string {
	return fmt.Sprintf("QuantizedTensor(bits=%d shape=%v ratio=%.1fx)", q.bits, q.origShape, q.CompressionRatio())
}
func Int8MatMul(a, b *QuantizedTensor) *Tensor {
	return a.Dequantize().MatMul(b.Dequantize())
}
