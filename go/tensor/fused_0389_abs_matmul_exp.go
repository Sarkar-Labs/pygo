package tensor

import "fmt"

func Fused0389AbsMatmulExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_matmul_exp: nil operand")
	}
	r := x.MatMul(y)
	r = r.Abs()
	r = r.Exp()
	return r, nil
}
func Fused0389AbsMatmulExpName() string { return "abs_matmul_exp" }
