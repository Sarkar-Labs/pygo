package tensor

import "fmt"

func Fused1350SiluAddRelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_add_relu: nil operand")
	}
	r := x.Add(y)
	r = r.SiLU()
	r = r.ReLU()
	return r, nil
}
func Fused1350SiluAddReluName() string { return "silu_add_relu" }
