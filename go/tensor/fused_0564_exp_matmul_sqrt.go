package tensor

import "fmt"

func Fused0564ExpMatmulSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_matmul_sqrt: nil operand")
	}
	r := x.MatMul(y)
	r = r.Exp()
	r = r.Sqrt()
	return r, nil
}
func Fused0564ExpMatmulSqrtName() string { return "exp_matmul_sqrt" }
