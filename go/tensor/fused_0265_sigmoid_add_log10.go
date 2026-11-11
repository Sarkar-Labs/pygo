package tensor

import "fmt"

func Fused0265SigmoidAddLog10(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_add_log10: nil operand")
	}
	r := x.Add(y)
	r = r.Sigmoid()
	r = r.Log10()
	return r, nil
}
func Fused0265SigmoidAddLog10Name() string { return "sigmoid_add_log10" }
