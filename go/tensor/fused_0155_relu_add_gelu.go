package tensor

import "fmt"

func Fused0155ReluAddGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_add_gelu: nil operand")
	}
	r := x.Add(y)
	r = r.ReLU()
	r = r.GELU()
	return r, nil
}
func Fused0155ReluAddGeluName() string { return "relu_add_gelu" }
