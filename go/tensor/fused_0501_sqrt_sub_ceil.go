package tensor

import "fmt"

func Fused0501SqrtSubCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_sub_ceil: nil operand")
	}
	r := x.Sub(y)
	r = r.Sqrt()
	r = r.Ceil()
	return r, nil
}
func Fused0501SqrtSubCeilName() string { return "sqrt_sub_ceil" }
