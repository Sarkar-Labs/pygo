package tensor

import "fmt"

func Fused0683LogDivCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_div_ceil: nil operand")
	}
	r := x.Div(y)
	r = r.Log()
	r = r.Ceil()
	return r, nil
}
func Fused0683LogDivCeilName() string { return "log_div_ceil" }
