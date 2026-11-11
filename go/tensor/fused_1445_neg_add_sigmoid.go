package tensor

import "fmt"

func Fused1445NegAddSigmoid(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_add_sigmoid: nil operand")
	}
	r := x.Add(y)
	r = r.Neg()
	r = r.Sigmoid()
	return r, nil
}
func Fused1445NegAddSigmoidName() string { return "neg_add_sigmoid" }
