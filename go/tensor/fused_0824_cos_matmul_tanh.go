package tensor

import "fmt"

func Fused0824CosMatmulTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_matmul_tanh: nil operand")
	}
	r := x.MatMul(y)
	r = r.Cos()
	r = r.Tanh()
	return r, nil
}
func Fused0824CosMatmulTanhName() string { return "cos_matmul_tanh" }
