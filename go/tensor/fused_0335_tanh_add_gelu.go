package tensor

import "fmt"

func Fused0335TanhAddGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_add_gelu: nil operand")
	}
	r := x.Add(y)
	r = r.Tanh()
	r = r.GELU()
	return r, nil
}
func Fused0335TanhAddGeluName() string { return "tanh_add_gelu" }
