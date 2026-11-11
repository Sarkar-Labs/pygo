package tensor

import "fmt"

func Fused1020CeilAddLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_add_log: nil operand")
	}
	r := x.Add(y)
	r = r.Ceil()
	r = r.Log()
	return r, nil
}
func Fused1020CeilAddLogName() string { return "ceil_add_log" }
