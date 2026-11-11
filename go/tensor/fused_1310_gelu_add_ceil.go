package tensor

import "fmt"

func Fused1310GeluAddCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_add_ceil: nil operand")
	}
	r := x.Add(y)
	r = r.GELU()
	r = r.Ceil()
	return r, nil
}
func Fused1310GeluAddCeilName() string { return "gelu_add_ceil" }
