package tensor

import "fmt"

func Fused0721SinSubRelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_sub_relu: nil operand")
	}
	r := x.Sub(y)
	r = r.Sin()
	r = r.ReLU()
	return r, nil
}
func Fused0721SinSubReluName() string { return "sin_sub_relu" }
