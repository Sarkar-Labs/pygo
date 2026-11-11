package tensor

import "fmt"

func Fused1651Log10SubLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_sub_log: nil operand")
	}
	r := x.Sub(y)
	r = r.Log10()
	r = r.Log()
	return r, nil
}
func Fused1651Log10SubLogName() string { return "log10_sub_log" }
