package tensor

import "fmt"

func Fused0738SinDivAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_div_abs: nil operand")
	}
	r := x.Div(y)
	r = r.Sin()
	r = r.Abs()
	return r, nil
}
func Fused0738SinDivAbsName() string { return "sin_div_abs" }
