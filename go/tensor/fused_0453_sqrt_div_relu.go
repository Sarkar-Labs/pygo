package tensor

import "fmt"

func Fused0453SqrtDivRelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_div_relu: nil operand")
	}
	r := x.Div(y)
	r = r.Sqrt()
	r = r.ReLU()
	return r, nil
}
func Fused0453SqrtDivReluName() string { return "sqrt_div_relu" }
