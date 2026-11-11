package tensor

import "fmt"

func Fused0515SqrtAddGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_add_gelu: nil operand")
	}
	r := x.Add(y)
	r = r.Sqrt()
	r = r.GELU()
	return r, nil
}
func Fused0515SqrtAddGeluName() string { return "sqrt_add_gelu" }
