package tensor

import "fmt"

func Fused1621Log10SubRelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_sub_relu: nil operand")
	}
	r := x.Sub(y)
	r = r.Log10()
	r = r.ReLU()
	return r, nil
}
func Fused1621Log10SubReluName() string { return "log10_sub_relu" }
