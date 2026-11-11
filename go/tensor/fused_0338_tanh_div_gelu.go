package tensor

import "fmt"

func Fused0338TanhDivGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_div_gelu: nil operand")
	}
	r := x.Div(y)
	r = r.Tanh()
	r = r.GELU()
	return r, nil
}
func Fused0338TanhDivGeluName() string { return "tanh_div_gelu" }
