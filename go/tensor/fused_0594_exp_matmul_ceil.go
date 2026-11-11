package tensor

import "fmt"

func Fused0594ExpMatmulCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_matmul_ceil: nil operand")
	}
	r := x.MatMul(y)
	r = r.Exp()
	r = r.Ceil()
	return r, nil
}
func Fused0594ExpMatmulCeilName() string { return "exp_matmul_ceil" }
