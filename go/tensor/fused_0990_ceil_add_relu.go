package tensor

import "fmt"

func Fused0990CeilAddRelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_add_relu: nil operand")
	}
	r := x.Add(y)
	r = r.Ceil()
	r = r.ReLU()
	return r, nil
}
func Fused0990CeilAddReluName() string { return "ceil_add_relu" }
