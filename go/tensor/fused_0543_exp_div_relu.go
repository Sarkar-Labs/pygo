package tensor

import "fmt"

func Fused0543ExpDivRelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_div_relu: nil operand")
	}
	r := x.Div(y)
	r = r.Exp()
	r = r.ReLU()
	return r, nil
}
func Fused0543ExpDivReluName() string { return "exp_div_relu" }
