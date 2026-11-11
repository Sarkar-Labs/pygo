package tensor

import "fmt"

func Fused0391AbsSubLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_sub_log: nil operand")
	}
	r := x.Sub(y)
	r = r.Abs()
	r = r.Log()
	return r, nil
}
func Fused0391AbsSubLogName() string { return "abs_sub_log" }
