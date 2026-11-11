package tensor

import "fmt"

func Fused0374AbsMatmulTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_matmul_tanh: nil operand")
	}
	r := x.MatMul(y)
	r = r.Abs()
	r = r.Tanh()
	return r, nil
}
func Fused0374AbsMatmulTanhName() string { return "abs_matmul_tanh" }
