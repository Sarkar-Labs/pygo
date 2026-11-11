package tensor

import "fmt"

func Fused1705Log10AddLog10(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_add_log10: nil operand")
	}
	r := x.Add(y)
	r = r.Log10()
	r = r.Log10()
	return r, nil
}
func Fused1705Log10AddLog10Name() string { return "log10_add_log10" }
