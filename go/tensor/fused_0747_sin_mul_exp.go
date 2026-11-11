package tensor

import "fmt"

func Fused0747SinMulExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_mul_exp: nil operand")
	}
	r := x.Mul(y)
	r = r.Sin()
	r = r.Exp()
	return r, nil
}
func Fused0747SinMulExpName() string { return "sin_mul_exp" }
