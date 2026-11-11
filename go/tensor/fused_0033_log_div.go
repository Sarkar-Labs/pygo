package tensor

import "fmt"

func Fused0033LogDiv(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_div: nil operand")
	}
	r := x.Div(y)
	r = r.Log()
	return r, nil
}
func Fused0033LogDivName() string { return "log_div" }
