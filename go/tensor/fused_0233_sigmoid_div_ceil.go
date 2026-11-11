package tensor

import "fmt"

func Fused0233SigmoidDivCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_div_ceil: nil operand")
	}
	r := x.Div(y)
	r = r.Sigmoid()
	r = r.Ceil()
	return r, nil
}
func Fused0233SigmoidDivCeilName() string { return "sigmoid_div_ceil" }
