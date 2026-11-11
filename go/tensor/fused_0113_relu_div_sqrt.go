package tensor

import "fmt"

func Fused0113ReluDivSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_div_sqrt: nil operand")
	}
	r := x.Div(y)
	r = r.ReLU()
	r = r.Sqrt()
	return r, nil
}
func Fused0113ReluDivSqrtName() string { return "relu_div_sqrt" }
