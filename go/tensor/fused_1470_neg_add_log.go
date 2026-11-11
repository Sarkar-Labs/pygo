package tensor

import "fmt"

func Fused1470NegAddLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_add_log: nil operand")
	}
	r := x.Add(y)
	r = r.Neg()
	r = r.Log()
	return r, nil
}
func Fused1470NegAddLogName() string { return "neg_add_log" }
