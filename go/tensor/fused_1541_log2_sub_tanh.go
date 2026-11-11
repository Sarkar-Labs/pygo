package tensor

import "fmt"

func Fused1541Log2SubTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_sub_tanh: nil operand")
	}
	r := x.Sub(y)
	r = r.Log2()
	r = r.Tanh()
	return r, nil
}
func Fused1541Log2SubTanhName() string { return "log2_sub_tanh" }
