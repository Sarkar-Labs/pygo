package tensor

import "fmt"

func Fused0502SqrtMulCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_mul_ceil: nil operand")
	}
	r := x.Mul(y)
	r = r.Sqrt()
	r = r.Ceil()
	return r, nil
}
func Fused0502SqrtMulCeilName() string { return "sqrt_mul_ceil" }
