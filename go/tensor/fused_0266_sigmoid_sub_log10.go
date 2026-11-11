package tensor

import "fmt"

func Fused0266SigmoidSubLog10(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_sub_log10: nil operand")
	}
	r := x.Sub(y)
	r = r.Sigmoid()
	r = r.Log10()
	return r, nil
}
func Fused0266SigmoidSubLog10Name() string { return "sigmoid_sub_log10" }
