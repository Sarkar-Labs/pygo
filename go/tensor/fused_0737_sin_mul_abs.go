package tensor

import "fmt"

func Fused0737SinMulAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_mul_abs: nil operand")
	}
	r := x.Mul(y)
	r = r.Sin()
	r = r.Abs()
	return r, nil
}
func Fused0737SinMulAbsName() string { return "sin_mul_abs" }
