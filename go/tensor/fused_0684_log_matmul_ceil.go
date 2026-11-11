package tensor

import "fmt"

func Fused0684LogMatmulCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_matmul_ceil: nil operand")
	}
	r := x.MatMul(y)
	r = r.Log()
	r = r.Ceil()
	return r, nil
}
func Fused0684LogMatmulCeilName() string { return "log_matmul_ceil" }
