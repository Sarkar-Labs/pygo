package tensor

import "fmt"

func Fused0288TanhDivAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_div_abs: nil operand")
	}
	r := x.Div(y)
	r = r.Tanh()
	r = r.Abs()
	return r, nil
}
func Fused0288TanhDivAbsName() string { return "tanh_div_abs" }
