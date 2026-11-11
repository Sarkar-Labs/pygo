package tensor

import "fmt"

func Fused1510NegAddSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_add_silu: nil operand")
	}
	r := x.Add(y)
	r = r.Neg()
	r = r.SiLU()
	return r, nil
}
func Fused1510NegAddSiluName() string { return "neg_add_silu" }
