package tensor

import "fmt"

func Fused0579ExpMatmulSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_matmul_sin: nil operand")
	}
	r := x.MatMul(y)
	r = r.Exp()
	r = r.Sin()
	return r, nil
}
func Fused0579ExpMatmulSinName() string { return "exp_matmul_sin" }
