package tensor

import "fmt"

func Fused0253SigmoidDivSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_div_silu: nil operand")
	}
	r := x.Div(y)
	r = r.Sigmoid()
	r = r.SiLU()
	return r, nil
}
func Fused0253SigmoidDivSiluName() string { return "sigmoid_div_silu" }
