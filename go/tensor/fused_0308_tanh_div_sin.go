package tensor

import "fmt"

func Fused0308TanhDivSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_div_sin: nil operand")
	}
	r := x.Div(y)
	r = r.Tanh()
	r = r.Sin()
	return r, nil
}
func Fused0308TanhDivSinName() string { return "tanh_div_sin" }
