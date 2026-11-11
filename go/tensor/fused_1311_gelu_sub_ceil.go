package tensor

import "fmt"

func Fused1311GeluSubCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_sub_ceil: nil operand")
	}
	r := x.Sub(y)
	r = r.GELU()
	r = r.Ceil()
	return r, nil
}
func Fused1311GeluSubCeilName() string { return "gelu_sub_ceil" }
