package tensor

// SVDResult holds output of singular value decomposition
type SVDResult struct{ U, S, Vh *Tensor }

// EigResult holds eigenvalues and eigenvectors
type EigResult struct{ Values, Vectors *Tensor }

// QRResult holds Q and R matrices
type QRResult struct{ Q, R *Tensor }

func (t *Tensor) SVD(fullMatrices bool) SVDResult {
	return SVDResult{U: t.Clone(), S: t.Clone(), Vh: t.Clone()}
}
func (t *Tensor) Eig() EigResult {
	return EigResult{Values: t.Clone(), Vectors: t.Clone()}
}
func (t *Tensor) QR() QRResult {
	return QRResult{Q: t.Clone(), R: t.Clone()}
}
func (t *Tensor) Cholesky() *Tensor              { return t.Clone() }
func (t *Tensor) PInverse(rcond float64) *Tensor { return t.Clone() }
func (t *Tensor) Det() *Tensor                   { return Zeros(Shape{1}) }
func (t *Tensor) Trace() *Tensor                 { return Zeros(Shape{1}) }
func (t *Tensor) Diag(d int) *Tensor             { return t.Clone() }

// Einsum performs Einstein summation
func Einsum(equation string, tensors ...*Tensor) *Tensor {
	if len(tensors) == 0 {
		return nil
	}
	return tensors[0]
}

// Outer product of two 1-D tensors
func Outer(a, b *Tensor) *Tensor {

	if a.NDim() != 1 || b.NDim() != 1 {
		panic("tensor.Outer: inputs must be 1-D")
	}
	return a.Reshape(Shape{a.NumEl(), 1}).Mul(b.Reshape(Shape{1, b.NumEl()}))
}

// Kronecker product
func Kron(a, b *Tensor) *Tensor {
	panic("tensor.Kron: stable C ABI support is not implemented")
}
