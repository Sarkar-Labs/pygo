package tensor

import "fmt"

func Fused1540Log2AddTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_add_tanh: nil operand")
	}
	r := x.Add(y)
	r = r.Log2()
	r = r.Tanh()
	return r, nil
}
func Fused1540Log2AddTanhName() string { return "log2_add_tanh" }
