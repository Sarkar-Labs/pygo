package tensor

import "fmt"

func Fused0018AbsDiv(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_div: nil operand")
	}
	r := x.Div(y)
	r = r.Abs()
	return r, nil
}
func Fused0018AbsDivName() string { return "abs_div" }
