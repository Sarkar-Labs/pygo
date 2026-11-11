package tensor

import "fmt"

func Fused1706Log10SubLog10(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_sub_log10: nil operand")
	}
	r := x.Sub(y)
	r = r.Log10()
	r = r.Log10()
	return r, nil
}
func Fused1706Log10SubLog10Name() string { return "log10_sub_log10" }
