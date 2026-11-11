package tensor

import "fmt"

func Fused0379AbsMatmulAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_matmul_abs: nil operand")
	}
	r := x.MatMul(y)
	r = r.Abs()
	r = r.Abs()
	return r, nil
}
func Fused0379AbsMatmulAbsName() string { return "abs_matmul_abs" }
