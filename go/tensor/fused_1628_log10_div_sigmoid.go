package tensor

import "fmt"

func Fused1628Log10DivSigmoid(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_div_sigmoid: nil operand")
	}
	r := x.Div(y)
	r = r.Log10()
	r = r.Sigmoid()
	return r, nil
}
func Fused1628Log10DivSigmoidName() string { return "log10_div_sigmoid" }
