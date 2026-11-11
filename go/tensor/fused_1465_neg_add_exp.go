package tensor

import "fmt"

func Fused1465NegAddExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_add_exp: nil operand")
	}
	r := x.Add(y)
	r = r.Neg()
	r = r.Exp()
	return r, nil
}
func Fused1465NegAddExpName() string { return "neg_add_exp" }
