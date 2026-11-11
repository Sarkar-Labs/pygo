package tensor

import "fmt"

func Fused0828CosDivAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_div_abs: nil operand")
	}
	r := x.Div(y)
	r = r.Cos()
	r = r.Abs()
	return r, nil
}
func Fused0828CosDivAbsName() string { return "cos_div_abs" }
