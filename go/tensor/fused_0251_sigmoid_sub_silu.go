package tensor

import "fmt"

func Fused0251SigmoidSubSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_sub_silu: nil operand")
	}
	r := x.Sub(y)
	r = r.Sigmoid()
	r = r.SiLU()
	return r, nil
}
func Fused0251SigmoidSubSiluName() string { return "sigmoid_sub_silu" }
