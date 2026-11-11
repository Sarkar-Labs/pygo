package tensor

import "fmt"

func Fused0037SinMul(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_mul: nil operand")
	}
	r := x.Mul(y)
	r = r.Sin()
	return r, nil
}
func Fused0037SinMulName() string { return "sin_mul" }
