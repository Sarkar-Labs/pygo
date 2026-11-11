package tensor

import "fmt"

func Fused0518SqrtDivGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_div_gelu: nil operand")
	}
	r := x.Div(y)
	r = r.Sqrt()
	r = r.GELU()
	return r, nil
}
func Fused0518SqrtDivGeluName() string { return "sqrt_div_gelu" }
