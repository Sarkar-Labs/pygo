package tensor

import "fmt"

func Fused0255SigmoidAddNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_add_neg: nil operand")
	}
	r := x.Add(y)
	r = r.Sigmoid()
	r = r.Neg()
	return r, nil
}
func Fused0255SigmoidAddNegName() string { return "sigmoid_add_neg" }
