package tensor

import "fmt"

func Fused0298TanhDivExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_div_exp: nil operand")
	}
	r := x.Div(y)
	r = r.Tanh()
	r = r.Exp()
	return r, nil
}
func Fused0298TanhDivExpName() string { return "tanh_div_exp" }
