package tensor

import "fmt"

func Fused1193SignDivSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_div_sqrt: nil operand")
	}
	r := x.Div(y)
	r = r.Sign()
	r = r.Sqrt()
	return r, nil
}
func Fused1193SignDivSqrtName() string { return "sign_div_sqrt" }
