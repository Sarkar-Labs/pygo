package tensor

import "fmt"

func Fused1021CeilSubLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_sub_log: nil operand")
	}
	r := x.Sub(y)
	r = r.Ceil()
	r = r.Log()
	return r, nil
}
func Fused1021CeilSubLogName() string { return "ceil_sub_log" }
