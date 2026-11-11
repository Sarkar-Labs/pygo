package tensor

import "fmt"

func Fused1560Log2AddLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_add_log: nil operand")
	}
	r := x.Add(y)
	r = r.Log2()
	r = r.Log()
	return r, nil
}
func Fused1560Log2AddLogName() string { return "log2_add_log" }
