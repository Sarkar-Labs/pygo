package tensor

import "fmt"

func Fused0160ReluAddSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_add_silu: nil operand")
	}
	r := x.Add(y)
	r = r.ReLU()
	r = r.SiLU()
	return r, nil
}
func Fused0160ReluAddSiluName() string { return "relu_add_silu" }
