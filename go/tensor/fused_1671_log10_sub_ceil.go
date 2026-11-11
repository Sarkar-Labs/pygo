package tensor

import "fmt"

func Fused1671Log10SubCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_sub_ceil: nil operand")
	}
	r := x.Sub(y)
	r = r.Log10()
	r = r.Ceil()
	return r, nil
}
func Fused1671Log10SubCeilName() string { return "log10_sub_ceil" }
