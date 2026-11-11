package tensor

import "fmt"

func Fused1325GeluAddGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_add_gelu: nil operand")
	}
	r := x.Add(y)
	r = r.GELU()
	r = r.GELU()
	return r, nil
}
func Fused1325GeluAddGeluName() string { return "gelu_add_gelu" }
