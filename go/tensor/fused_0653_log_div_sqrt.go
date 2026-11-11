package tensor

import "fmt"

func Fused0653LogDivSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_div_sqrt: nil operand")
	}
	r := x.Div(y)
	r = r.Log()
	r = r.Sqrt()
	return r, nil
}
func Fused0653LogDivSqrtName() string { return "log_div_sqrt" }
