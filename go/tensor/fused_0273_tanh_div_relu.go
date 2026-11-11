package tensor

import "fmt"

func Fused0273TanhDivRelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_div_relu: nil operand")
	}
	r := x.Div(y)
	r = r.Tanh()
	r = r.ReLU()
	return r, nil
}
func Fused0273TanhDivReluName() string { return "tanh_div_relu" }
