package tensor

import "fmt"

func Fused1455NegAddAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_add_abs: nil operand")
	}
	r := x.Add(y)
	r = r.Neg()
	r = r.Abs()
	return r, nil
}
func Fused1455NegAddAbsName() string { return "neg_add_abs" }
