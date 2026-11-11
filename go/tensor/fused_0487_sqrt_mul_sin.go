package tensor

import "fmt"

func Fused0487SqrtMulSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_mul_sin: nil operand")
	}
	r := x.Mul(y)
	r = r.Sqrt()
	r = r.Sin()
	return r, nil
}
func Fused0487SqrtMulSinName() string { return "sqrt_mul_sin" }
