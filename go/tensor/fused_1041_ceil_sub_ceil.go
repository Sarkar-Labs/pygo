package tensor

import "fmt"

func Fused1041CeilSubCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_sub_ceil: nil operand")
	}
	r := x.Sub(y)
	r = r.Ceil()
	r = r.Ceil()
	return r, nil
}
func Fused1041CeilSubCeilName() string { return "ceil_sub_ceil" }
