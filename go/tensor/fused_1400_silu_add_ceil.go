package tensor

import "fmt"

func Fused1400SiluAddCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_add_ceil: nil operand")
	}
	r := x.Add(y)
	r = r.SiLU()
	r = r.Ceil()
	return r, nil
}
func Fused1400SiluAddCeilName() string { return "silu_add_ceil" }
