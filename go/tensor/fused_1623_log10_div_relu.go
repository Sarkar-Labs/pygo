package tensor

import "fmt"

func Fused1623Log10DivRelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_div_relu: nil operand")
	}
	r := x.Div(y)
	r = r.Log10()
	r = r.ReLU()
	return r, nil
}
func Fused1623Log10DivReluName() string { return "log10_div_relu" }
