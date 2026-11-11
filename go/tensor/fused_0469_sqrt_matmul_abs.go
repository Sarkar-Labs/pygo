package tensor

import "fmt"

func Fused0469SqrtMatmulAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_matmul_abs: nil operand")
	}
	r := x.MatMul(y)
	r = r.Sqrt()
	r = r.Abs()
	return r, nil
}
func Fused0469SqrtMatmulAbsName() string { return "sqrt_matmul_abs" }
