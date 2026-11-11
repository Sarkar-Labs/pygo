package tensor

import "fmt"

func Fused1650Log10AddLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_add_log: nil operand")
	}
	r := x.Add(y)
	r = r.Log10()
	r = r.Log()
	return r, nil
}
func Fused1650Log10AddLogName() string { return "log10_add_log" }
