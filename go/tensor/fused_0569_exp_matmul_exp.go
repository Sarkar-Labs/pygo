package tensor

import "fmt"

func Fused0569ExpMatmulExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_matmul_exp: nil operand")
	}
	r := x.MatMul(y)
	r = r.Exp()
	r = r.Exp()
	return r, nil
}
func Fused0569ExpMatmulExpName() string { return "exp_matmul_exp" }
