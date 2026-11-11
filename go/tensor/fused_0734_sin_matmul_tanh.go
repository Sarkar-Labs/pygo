package tensor

import "fmt"

func Fused0734SinMatmulTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_matmul_tanh: nil operand")
	}
	r := x.MatMul(y)
	r = r.Sin()
	r = r.Tanh()
	return r, nil
}
func Fused0734SinMatmulTanhName() string { return "sin_matmul_tanh" }
