package tensor

import "fmt"

func Fused0393AbsDivLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_div_log: nil operand")
	}
	r := x.Div(y)
	r = r.Abs()
	r = r.Log()
	return r, nil
}
func Fused0393AbsDivLogName() string { return "abs_div_log" }
