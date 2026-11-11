package tensor

import "fmt"

func Fused0425AbsAddGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_add_gelu: nil operand")
	}
	r := x.Add(y)
	r = r.Abs()
	r = r.GELU()
	return r, nil
}
func Fused0425AbsAddGeluName() string { return "abs_add_gelu" }
