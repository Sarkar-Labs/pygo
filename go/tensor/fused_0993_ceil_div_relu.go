package tensor

import "fmt"

func Fused0993CeilDivRelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_div_relu: nil operand")
	}
	r := x.Div(y)
	r = r.Ceil()
	r = r.ReLU()
	return r, nil
}
func Fused0993CeilDivReluName() string { return "ceil_div_relu" }
