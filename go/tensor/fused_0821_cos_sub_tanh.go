package tensor

import "fmt"

func Fused0821CosSubTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_sub_tanh: nil operand")
	}
	r := x.Sub(y)
	r = r.Cos()
	r = r.Tanh()
	return r, nil
}
func Fused0821CosSubTanhName() string { return "cos_sub_tanh" }
