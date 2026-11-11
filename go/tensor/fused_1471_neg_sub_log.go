package tensor

import "fmt"

func Fused1471NegSubLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_sub_log: nil operand")
	}
	r := x.Sub(y)
	r = r.Neg()
	r = r.Log()
	return r, nil
}
func Fused1471NegSubLogName() string { return "neg_sub_log" }
