package tensor

import "fmt"

func Fused0091ReluSubRelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_sub_relu: nil operand")
	}
	r := x.Sub(y)
	r = r.ReLU()
	r = r.ReLU()
	return r, nil
}
func Fused0091ReluSubReluName() string { return "relu_sub_relu" }
