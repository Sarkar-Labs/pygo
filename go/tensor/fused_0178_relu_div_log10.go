package tensor

import "fmt"

func Fused0178ReluDivLog10(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_div_log10: nil operand")
	}
	r := x.Div(y)
	r = r.ReLU()
	r = r.Log10()
	return r, nil
}
func Fused0178ReluDivLog10Name() string { return "relu_div_log10" }
