package tensor

import "fmt"

func Fused0270TanhAddRelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_add_relu: nil operand")
	}
	r := x.Add(y)
	r = r.Tanh()
	r = r.ReLU()
	return r, nil
}
func Fused0270TanhAddReluName() string { return "tanh_add_relu" }
