package tensor

import "fmt"

func Fused1443NegDivRelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_div_relu: nil operand")
	}
	r := x.Div(y)
	r = r.Neg()
	r = r.ReLU()
	return r, nil
}
func Fused1443NegDivReluName() string { return "neg_div_relu" }
