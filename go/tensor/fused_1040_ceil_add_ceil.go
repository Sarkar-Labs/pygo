package tensor

import "fmt"

func Fused1040CeilAddCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_add_ceil: nil operand")
	}
	r := x.Add(y)
	r = r.Ceil()
	r = r.Ceil()
	return r, nil
}
func Fused1040CeilAddCeilName() string { return "ceil_add_ceil" }
