package tensor

import "fmt"

func Fused1055CeilAddGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_add_gelu: nil operand")
	}
	r := x.Add(y)
	r = r.Ceil()
	r = r.GELU()
	return r, nil
}
func Fused1055CeilAddGeluName() string { return "ceil_add_gelu" }
