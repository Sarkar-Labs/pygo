package tensor

import "fmt"

func Fused0771SinSubCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_sub_ceil: nil operand")
	}
	r := x.Sub(y)
	r = r.Sin()
	r = r.Ceil()
	return r, nil
}
func Fused0771SinSubCeilName() string { return "sin_sub_ceil" }
