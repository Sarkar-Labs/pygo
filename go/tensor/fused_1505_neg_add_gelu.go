package tensor

import "fmt"

func Fused1505NegAddGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_add_gelu: nil operand")
	}
	r := x.Add(y)
	r = r.Neg()
	r = r.GELU()
	return r, nil
}
func Fused1505NegAddGeluName() string { return "neg_add_gelu" }
