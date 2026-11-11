package tensor

import "fmt"

func Fused0813CosDivRelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_div_relu: nil operand")
	}
	r := x.Div(y)
	r = r.Cos()
	r = r.ReLU()
	return r, nil
}
func Fused0813CosDivReluName() string { return "cos_div_relu" }
