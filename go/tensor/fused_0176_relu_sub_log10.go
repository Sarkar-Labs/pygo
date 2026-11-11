package tensor

import "fmt"

func Fused0176ReluSubLog10(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_sub_log10: nil operand")
	}
	r := x.Sub(y)
	r = r.ReLU()
	r = r.Log10()
	return r, nil
}
func Fused0176ReluSubLog10Name() string { return "relu_sub_log10" }
