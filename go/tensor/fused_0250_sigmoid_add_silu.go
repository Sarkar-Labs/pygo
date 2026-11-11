package tensor

import "fmt"

func Fused0250SigmoidAddSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_add_silu: nil operand")
	}
	r := x.Add(y)
	r = r.Sigmoid()
	r = r.SiLU()
	return r, nil
}
func Fused0250SigmoidAddSiluName() string { return "sigmoid_add_silu" }
