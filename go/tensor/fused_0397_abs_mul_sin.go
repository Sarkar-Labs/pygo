package tensor

import "fmt"

func Fused0397AbsMulSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_mul_sin: nil operand")
	}
	r := x.Mul(y)
	r = r.Abs()
	r = r.Sin()
	return r, nil
}
func Fused0397AbsMulSinName() string { return "abs_mul_sin" }
