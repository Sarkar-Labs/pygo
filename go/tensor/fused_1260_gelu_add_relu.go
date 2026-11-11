package tensor

import "fmt"

func Fused1260GeluAddRelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_add_relu: nil operand")
	}
	r := x.Add(y)
	r = r.GELU()
	r = r.ReLU()
	return r, nil
}
func Fused1260GeluAddReluName() string { return "gelu_add_relu" }
