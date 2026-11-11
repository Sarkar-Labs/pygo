package tensor

import "fmt"

func Fused0256SigmoidSubNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_sub_neg: nil operand")
	}
	r := x.Sub(y)
	r = r.Sigmoid()
	r = r.Neg()
	return r, nil
}
func Fused0256SigmoidSubNegName() string { return "sigmoid_sub_neg" }
