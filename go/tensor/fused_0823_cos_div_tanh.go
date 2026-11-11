package tensor

import "fmt"

func Fused0823CosDivTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_div_tanh: nil operand")
	}
	r := x.Div(y)
	r = r.Cos()
	r = r.Tanh()
	return r, nil
}
func Fused0823CosDivTanhName() string { return "cos_div_tanh" }
