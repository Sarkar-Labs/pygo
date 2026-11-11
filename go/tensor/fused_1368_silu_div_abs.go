package tensor

import "fmt"

func Fused1368SiluDivAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_div_abs: nil operand")
	}
	r := x.Div(y)
	r = r.SiLU()
	r = r.Abs()
	return r, nil
}
func Fused1368SiluDivAbsName() string { return "silu_div_abs" }
