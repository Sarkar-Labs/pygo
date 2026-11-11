package tensor

import "fmt"

func Fused1401SiluSubCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_sub_ceil: nil operand")
	}
	r := x.Sub(y)
	r = r.SiLU()
	r = r.Ceil()
	return r, nil
}
func Fused1401SiluSubCeilName() string { return "silu_sub_ceil" }
