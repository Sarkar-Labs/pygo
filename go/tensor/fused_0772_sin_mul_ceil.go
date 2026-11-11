package tensor

import "fmt"

func Fused0772SinMulCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_mul_ceil: nil operand")
	}
	r := x.Mul(y)
	r = r.Sin()
	r = r.Ceil()
	return r, nil
}
func Fused0772SinMulCeilName() string { return "sin_mul_ceil" }
