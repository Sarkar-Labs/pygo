package tensor

import "fmt"

func Fused1385SiluAddSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_add_sin: nil operand")
	}
	r := x.Add(y)
	r = r.SiLU()
	r = r.Sin()
	return r, nil
}
func Fused1385SiluAddSinName() string { return "silu_add_sin" }
