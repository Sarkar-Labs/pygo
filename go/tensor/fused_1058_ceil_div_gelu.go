package tensor

import "fmt"

func Fused1058CeilDivGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_div_gelu: nil operand")
	}
	r := x.Div(y)
	r = r.Ceil()
	r = r.GELU()
	return r, nil
}
func Fused1058CeilDivGeluName() string { return "ceil_div_gelu" }
