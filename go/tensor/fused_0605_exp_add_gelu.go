package tensor

import "fmt"

func Fused0605ExpAddGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_add_gelu: nil operand")
	}
	r := x.Add(y)
	r = r.Exp()
	r = r.GELU()
	return r, nil
}
func Fused0605ExpAddGeluName() string { return "exp_add_gelu" }
