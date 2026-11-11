package tensor

import "fmt"

func Fused0820CosAddTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_add_tanh: nil operand")
	}
	r := x.Add(y)
	r = r.Cos()
	r = r.Tanh()
	return r, nil
}
func Fused0820CosAddTanhName() string { return "cos_add_tanh" }
