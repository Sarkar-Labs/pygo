package tensor

import "fmt"

func Fused0090ReluAddRelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_add_relu: nil operand")
	}
	r := x.Add(y)
	r = r.ReLU()
	r = r.ReLU()
	return r, nil
}
func Fused0090ReluAddReluName() string { return "relu_add_relu" }
