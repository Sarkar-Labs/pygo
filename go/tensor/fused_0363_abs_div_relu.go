package tensor

import "fmt"

func Fused0363AbsDivRelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_div_relu: nil operand")
	}
	r := x.Div(y)
	r = r.Abs()
	r = r.ReLU()
	return r, nil
}
func Fused0363AbsDivReluName() string { return "abs_div_relu" }
