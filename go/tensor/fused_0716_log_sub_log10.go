package tensor

import "fmt"

func Fused0716LogSubLog10(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_sub_log10: nil operand")
	}
	r := x.Sub(y)
	r = r.Log()
	r = r.Log10()
	return r, nil
}
func Fused0716LogSubLog10Name() string { return "log_sub_log10" }
