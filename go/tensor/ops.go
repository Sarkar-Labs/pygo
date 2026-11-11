package tensor

// Convenience tensor constructors and operations that are not defined in tensor.go.

func ZerosLike(t *Tensor) *Tensor {
	return Zeros(t.Shape(), WithDType(t.DType()), WithDevice(t.Device()))
}
func OnesLike(t *Tensor) *Tensor {
	return Ones(t.Shape(), WithDType(t.DType()), WithDevice(t.Device()))
}
func RandLike(t *Tensor) *Tensor {
	return Randn(t.Shape(), WithDType(t.DType()), WithDevice(t.Device()))
}

func Cat(tensors []*Tensor, dim int) *Tensor {
	if len(tensors) == 0 {
		return nil
	}
	if len(tensors) == 1 {
		return tensors[0]
	}
	// The native C ABI currently exposes binary operations but not concatenation.
	// Fail explicitly rather than silently returning the first tensor.
	panic("tensor.Cat: concatenation is not exposed by the current native ABI")
}

func Stack(tensors []*Tensor, dim int) *Tensor {
	if len(tensors) == 0 {
		return nil
	}
	if len(tensors) == 1 {
		return tensors[0].Unsqueeze(dim)
	}
	panic("tensor.Stack: stacking multiple tensors is not exposed by the current native ABI")
}

func (t *Tensor) SubScalar(v float64) *Tensor { return t.AddScalar(-v) }
func (t *Tensor) DivScalar(v float64) *Tensor {
	if v == 0 {
		panic("tensor.DivScalar: division by zero")
	}
	return t.MulScalar(1 / v)
}
func (t *Tensor) Clamp(mn, mx float64) *Tensor {
	if mn > mx {
		panic("tensor.Clamp: min must not exceed max")
	}
	// Use the native elementwise operations; the CPU runtime is float32 today.
	out := t.Clone()
	data := out.Float32Slice()
	for i, x := range data {
		if float64(x) < mn {
			data[i] = float32(mn)
		}
		if float64(data[i]) > mx {
			data[i] = float32(mx)
		}
	}
	return out
}
func (t *Tensor) Relu6() *Tensor { return t.Clamp(0, 6) }

func (t *Tensor) Slice(dim, start, end, step int64) *Tensor {
	return t.nativeSlice(int(dim), start, end, step)
}
func (t *Tensor) Norm(p float64, dim int, keepdim bool) *Tensor {
	return t.nativeNorm(p, dim, keepdim)
}
func (t *Tensor) Triu(diagonal int) *Tensor {
	return t.nativeTriu(diagonal)
}
func (t *Tensor) Index(idx *Tensor) *Tensor           { return t.nativeIndex(idx) }
func (t *Tensor) Gather(dim int, idx *Tensor) *Tensor { return t.nativeGather(dim, idx) }
func (t *Tensor) Expand(sizes ...int64) *Tensor       { return t.nativeExpand(sizes...) }

func (t *Tensor) Float32Slice() []float32 { return t.nativeFloat32Slice() }
func (t *Tensor) Int64Slice() []int64     { return t.nativeInt64Slice() }

func (t *Tensor) Sign() *Tensor                { return t.nativeSign() }
func (t *Tensor) Repeat(repeats []int) *Tensor { return t.nativeRepeat(repeats) }
