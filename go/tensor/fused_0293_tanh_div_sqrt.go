package tensor

import "fmt"

func Fused0293TanhDivSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_div_sqrt: nil operand")
	}
	r := x.Div(y)
	r = r.Tanh()
	r = r.Sqrt()
	return r, nil
}
func Fused0293TanhDivSqrtName() string { return "tanh_div_sqrt" }
