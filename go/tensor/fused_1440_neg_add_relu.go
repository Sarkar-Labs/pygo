package tensor

import "fmt"

func Fused1440NegAddRelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_add_relu: nil operand")
	}
	r := x.Add(y)
	r = r.Neg()
	r = r.ReLU()
	return r, nil
}
func Fused1440NegAddReluName() string { return "neg_add_relu" }
