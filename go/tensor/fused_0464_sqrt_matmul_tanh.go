package tensor

import "fmt"

func Fused0464SqrtMatmulTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_matmul_tanh: nil operand")
	}
	r := x.MatMul(y)
	r = r.Sqrt()
	r = r.Tanh()
	return r, nil
}
func Fused0464SqrtMatmulTanhName() string { return "sqrt_matmul_tanh" }
