package tensor

import "fmt"

func Fused1620Log10AddRelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_add_relu: nil operand")
	}
	r := x.Add(y)
	r = r.Log10()
	r = r.ReLU()
	return r, nil
}
func Fused1620Log10AddReluName() string { return "log10_add_relu" }
