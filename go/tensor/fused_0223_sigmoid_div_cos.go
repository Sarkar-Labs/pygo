package tensor

import "fmt"

func Fused0223SigmoidDivCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_div_cos: nil operand")
	}
	r := x.Div(y)
	r = r.Sigmoid()
	r = r.Cos()
	return r, nil
}
func Fused0223SigmoidDivCosName() string { return "sigmoid_div_cos" }
