package tensor

import "fmt"

func Fused1508NegDivGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_div_gelu: nil operand")
	}
	r := x.Div(y)
	r = r.Neg()
	r = r.GELU()
	return r, nil
}
func Fused1508NegDivGeluName() string { return "neg_div_gelu" }
