package tensor

import "fmt"

func Fused0559ExpMatmulAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_matmul_abs: nil operand")
	}
	r := x.MatMul(y)
	r = r.Exp()
	r = r.Abs()
	return r, nil
}
func Fused0559ExpMatmulAbsName() string { return "exp_matmul_abs" }
