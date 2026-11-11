package tensor

import "fmt"

func Fused1548Log2DivAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_div_abs: nil operand")
	}
	r := x.Div(y)
	r = r.Log2()
	r = r.Abs()
	return r, nil
}
func Fused1548Log2DivAbsName() string { return "log2_div_abs" }
