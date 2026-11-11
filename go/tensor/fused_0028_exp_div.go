package tensor

import "fmt"

func Fused0028ExpDiv(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_div: nil operand")
	}
	r := x.Div(y)
	r = r.Exp()
	return r, nil
}
func Fused0028ExpDivName() string { return "exp_div" }
