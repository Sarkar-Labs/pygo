package tensor

import "fmt"

func Fused1475NegAddSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_add_sin: nil operand")
	}
	r := x.Add(y)
	r = r.Neg()
	r = r.Sin()
	return r, nil
}
func Fused1475NegAddSinName() string { return "neg_add_sin" }
