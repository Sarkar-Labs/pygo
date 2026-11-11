package tensor

import "fmt"

func Fused0168ReluDivNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_div_neg: nil operand")
	}
	r := x.Div(y)
	r = r.ReLU()
	r = r.Neg()
	return r, nil
}
func Fused0168ReluDivNegName() string { return "relu_div_neg" }
