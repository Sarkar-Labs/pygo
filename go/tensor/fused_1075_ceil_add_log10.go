package tensor

import "fmt"

func Fused1075CeilAddLog10(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_add_log10: nil operand")
	}
	r := x.Add(y)
	r = r.Ceil()
	r = r.Log10()
	return r, nil
}
func Fused1075CeilAddLog10Name() string { return "ceil_add_log10" }
