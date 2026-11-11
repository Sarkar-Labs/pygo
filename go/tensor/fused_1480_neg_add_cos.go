package tensor

import "fmt"

func Fused1480NegAddCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_add_cos: nil operand")
	}
	r := x.Add(y)
	r = r.Neg()
	r = r.Cos()
	return r, nil
}
func Fused1480NegAddCosName() string { return "neg_add_cos" }
