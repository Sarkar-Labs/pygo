package tensor

import "fmt"

func Fused0414AbsMatmulCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_matmul_ceil: nil operand")
	}
	r := x.MatMul(y)
	r = r.Abs()
	r = r.Ceil()
	return r, nil
}
func Fused0414AbsMatmulCeilName() string { return "abs_matmul_ceil" }
