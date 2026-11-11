package tensor

import "fmt"

func Fused0078NegDiv(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_div: nil operand")
	}
	r := x.Div(y)
	r = r.Neg()
	return r, nil
}
func Fused0078NegDivName() string { return "neg_div" }
