package tensor

import "fmt"

func Fused1672Log10MulCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_mul_ceil: nil operand")
	}
	r := x.Mul(y)
	r = r.Log10()
	r = r.Ceil()
	return r, nil
}
func Fused1672Log10MulCeilName() string { return "log10_mul_ceil" }
