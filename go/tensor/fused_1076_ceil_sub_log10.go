package tensor

import "fmt"

func Fused1076CeilSubLog10(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_sub_log10: nil operand")
	}
	r := x.Sub(y)
	r = r.Ceil()
	r = r.Log10()
	return r, nil
}
func Fused1076CeilSubLog10Name() string { return "ceil_sub_log10" }
