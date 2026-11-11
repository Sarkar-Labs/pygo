package tensor

import "fmt"

func Fused0649LogMatmulAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_matmul_abs: nil operand")
	}
	r := x.MatMul(y)
	r = r.Log()
	r = r.Abs()
	return r, nil
}
func Fused0649LogMatmulAbsName() string { return "log_matmul_abs" }
