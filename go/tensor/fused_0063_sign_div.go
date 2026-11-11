package tensor

import "fmt"

func Fused0063SignDiv(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_div: nil operand")
	}
	r := x.Div(y)
	r = r.Sign()
	return r, nil
}
func Fused0063SignDivName() string { return "sign_div" }
