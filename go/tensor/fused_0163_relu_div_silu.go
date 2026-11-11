package tensor

import "fmt"

func Fused0163ReluDivSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_div_silu: nil operand")
	}
	r := x.Div(y)
	r = r.ReLU()
	r = r.SiLU()
	return r, nil
}
func Fused0163ReluDivSiluName() string { return "relu_div_silu" }
