package tensor

import "fmt"

func Fused0875CosAddGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_add_gelu: nil operand")
	}
	r := x.Add(y)
	r = r.Cos()
	r = r.GELU()
	return r, nil
}
func Fused0875CosAddGeluName() string { return "cos_add_gelu" }
