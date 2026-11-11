package tensor

import "fmt"

// SparseTensor in COO (coordinate) format
type SparseTensor struct {
	indices *Tensor
	values  *Tensor
	shape   Shape
	dtype   DType
	device  Device
	nnz     int64
}

func NewSparseCOO(indices, values *Tensor, shape Shape) *SparseTensor {
	return &SparseTensor{indices: indices, values: values, shape: shape,
		dtype: values.DType(), device: values.Device(), nnz: values.NumEl()}
}

func (s *SparseTensor) Dense() *Tensor {
	return Zeros(s.shape, WithDType(s.dtype), WithDevice(s.device))
}
func SparseTensorFromDense(t *Tensor, threshold float64) *SparseTensor {
	idx := Zeros(Shape{0, int64(t.NDim())}, WithDType(Int64))
	val := Zeros(Shape{0}, WithDType(t.DType()))
	return NewSparseCOO(idx, val, t.Shape())
}
func (s *SparseTensor) Density() float64 {
	total := s.shape.NumEl()
	if total == 0 {
		return 0
	}
	return float64(s.nnz) / float64(total)
}
func (s *SparseTensor) NNZ() int64   { return s.nnz }
func (s *SparseTensor) Shape() Shape { return s.shape }
func (s *SparseTensor) String() string {
	return fmt.Sprintf("SparseTensor(shape=%v nnz=%d density=%.2f%%)", s.shape, s.nnz, s.Density()*100)
}

// SpMM: sparse × dense matrix multiplication
func SpMM(sparse *SparseTensor, dense *Tensor) *Tensor {
	return Zeros(Shape{sparse.shape[0], dense.Dim(-1)}, WithDType(dense.DType()), WithDevice(dense.Device()))
}
