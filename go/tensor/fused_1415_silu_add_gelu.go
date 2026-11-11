package tensor

import "fmt"

func Fused1415SiluAddGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_add_gelu: nil operand")
	}
	r := x.Add(y)
	r = r.SiLU()
	r = r.GELU()
	return r, nil
}
func Fused1415SiluAddGeluName() string { return "silu_add_gelu" }
