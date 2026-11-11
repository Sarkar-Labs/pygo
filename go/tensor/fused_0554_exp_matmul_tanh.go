package tensor

import "fmt"

func Fused0554ExpMatmulTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_matmul_tanh: nil operand")
	}
	r := x.MatMul(y)
	r = r.Exp()
	r = r.Tanh()
	return r, nil
}
func Fused0554ExpMatmulTanhName() string { return "exp_matmul_tanh" }
