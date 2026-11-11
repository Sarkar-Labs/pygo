package tensor

import "fmt"

func Fused0208SigmoidDivExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_div_exp: nil operand")
	}
	r := x.Div(y)
	r = r.Sigmoid()
	r = r.Exp()
	return r, nil
}
func Fused0208SigmoidDivExpName() string { return "sigmoid_div_exp" }
