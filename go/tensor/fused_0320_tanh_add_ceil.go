package tensor

import "fmt"

func Fused0320TanhAddCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_add_ceil: nil operand")
	}
	r := x.Add(y)
	r = r.Tanh()
	r = r.Ceil()
	return r, nil
}
func Fused0320TanhAddCeilName() string { return "tanh_add_ceil" }
