package tensor

import "fmt"

func Fused0368AbsDivSigmoid(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_div_sigmoid: nil operand")
	}
	r := x.Div(y)
	r = r.Abs()
	r = r.Sigmoid()
	return r, nil
}
func Fused0368AbsDivSigmoidName() string { return "abs_div_sigmoid" }
