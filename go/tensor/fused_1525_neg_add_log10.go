package tensor

import "fmt"

func Fused1525NegAddLog10(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_add_log10: nil operand")
	}
	r := x.Add(y)
	r = r.Neg()
	r = r.Log10()
	return r, nil
}
func Fused1525NegAddLog10Name() string { return "neg_add_log10" }
