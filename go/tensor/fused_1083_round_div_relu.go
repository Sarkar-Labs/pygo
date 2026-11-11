package tensor

import "fmt"

func Fused1083RoundDivRelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_div_relu: nil operand")
	}
	r := x.Div(y)
	r = r.Round()
	r = r.ReLU()
	return r, nil
}
func Fused1083RoundDivReluName() string { return "round_div_relu" }
