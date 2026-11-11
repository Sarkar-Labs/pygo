package tensor

import "fmt"

func Fused0323TanhDivCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_div_ceil: nil operand")
	}
	r := x.Div(y)
	r = r.Tanh()
	r = r.Ceil()
	return r, nil
}
func Fused0323TanhDivCeilName() string { return "tanh_div_ceil" }
