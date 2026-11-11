package tensor

import "fmt"

func Fused0661LogSubLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_sub_log: nil operand")
	}
	r := x.Sub(y)
	r = r.Log()
	r = r.Log()
	return r, nil
}
func Fused0661LogSubLogName() string { return "log_sub_log" }
