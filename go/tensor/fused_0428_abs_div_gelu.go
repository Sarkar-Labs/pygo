package tensor

import "fmt"

func Fused0428AbsDivGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_div_gelu: nil operand")
	}
	r := x.Div(y)
	r = r.Abs()
	r = r.GELU()
	return r, nil
}
func Fused0428AbsDivGeluName() string { return "abs_div_gelu" }
