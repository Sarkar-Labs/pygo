package tensor

import "fmt"

func Fused0860CosAddCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_add_ceil: nil operand")
	}
	r := x.Add(y)
	r = r.Cos()
	r = r.Ceil()
	return r, nil
}
func Fused0860CosAddCeilName() string { return "cos_add_ceil" }
