package tensor

import "fmt"

func Fused1473NegDivLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_div_log: nil operand")
	}
	r := x.Div(y)
	r = r.Neg()
	r = r.Log()
	return r, nil
}
func Fused1473NegDivLogName() string { return "neg_div_log" }
