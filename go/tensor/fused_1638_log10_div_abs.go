package tensor

import "fmt"

func Fused1638Log10DivAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_div_abs: nil operand")
	}
	r := x.Div(y)
	r = r.Log10()
	r = r.Abs()
	return r, nil
}
func Fused1638Log10DivAbsName() string { return "log10_div_abs" }
