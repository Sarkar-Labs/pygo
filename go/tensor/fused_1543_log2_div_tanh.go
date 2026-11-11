package tensor

import "fmt"

func Fused1543Log2DivTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_div_tanh: nil operand")
	}
	r := x.Div(y)
	r = r.Log2()
	r = r.Tanh()
	return r, nil
}
func Fused1543Log2DivTanhName() string { return "log2_div_tanh" }
