package tensor

import "fmt"

func Fused0785SinAddGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_add_gelu: nil operand")
	}
	r := x.Add(y)
	r = r.Sin()
	r = r.GELU()
	return r, nil
}
func Fused0785SinAddGeluName() string { return "sin_add_gelu" }
