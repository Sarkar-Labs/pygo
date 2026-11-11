package tensor

import "fmt"

func Fused0023SqrtDiv(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_div: nil operand")
	}
	r := x.Div(y)
	r = r.Sqrt()
	return r, nil
}
func Fused0023SqrtDivName() string { return "sqrt_div" }
