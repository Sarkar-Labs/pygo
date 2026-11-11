package tensor

import "fmt"

func Fused0000ReluAdd(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_add: nil operand")
	}
	r := x.Add(y)
	r = r.ReLU()
	return r, nil
}
func Fused0000ReluAddName() string { return "relu_add" }
