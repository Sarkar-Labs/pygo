package tensor

import "fmt"

func Fused0774SinMatmulCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_matmul_ceil: nil operand")
	}
	r := x.MatMul(y)
	r = r.Sin()
	r = r.Ceil()
	return r, nil
}
func Fused0774SinMatmulCeilName() string { return "sin_matmul_ceil" }
