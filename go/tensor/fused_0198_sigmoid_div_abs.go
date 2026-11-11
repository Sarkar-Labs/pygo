package tensor

import "fmt"

func Fused0198SigmoidDivAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_div_abs: nil operand")
	}
	r := x.Div(y)
	r = r.Sigmoid()
	r = r.Abs()
	return r, nil
}
func Fused0198SigmoidDivAbsName() string { return "sigmoid_div_abs" }
