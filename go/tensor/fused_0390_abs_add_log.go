package tensor

import "fmt"

func Fused0390AbsAddLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_add_log: nil operand")
	}
	r := x.Add(y)
	r = r.Abs()
	r = r.Log()
	return r, nil
}
func Fused0390AbsAddLogName() string { return "abs_add_log" }
