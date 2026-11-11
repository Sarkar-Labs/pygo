package tensor

import "fmt"

func Fused0161ReluSubSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_sub_silu: nil operand")
	}
	r := x.Sub(y)
	r = r.ReLU()
	r = r.SiLU()
	return r, nil
}
func Fused0161ReluSubSiluName() string { return "relu_sub_silu" }
