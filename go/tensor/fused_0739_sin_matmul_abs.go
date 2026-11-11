package tensor

import "fmt"

func Fused0739SinMatmulAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_matmul_abs: nil operand")
	}
	r := x.MatMul(y)
	r = r.Sin()
	r = r.Abs()
	return r, nil
}
func Fused0739SinMatmulAbsName() string { return "sin_matmul_abs" }
