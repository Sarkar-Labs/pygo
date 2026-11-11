package tensor

import "fmt"

func Fused1490NegAddCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_add_ceil: nil operand")
	}
	r := x.Add(y)
	r = r.Neg()
	r = r.Ceil()
	return r, nil
}
func Fused1490NegAddCeilName() string { return "neg_add_ceil" }
