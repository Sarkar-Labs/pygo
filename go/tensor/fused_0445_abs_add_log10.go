package tensor

import "fmt"

func Fused0445AbsAddLog10(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_add_log10: nil operand")
	}
	r := x.Add(y)
	r = r.Abs()
	r = r.Log10()
	return r, nil
}
func Fused0445AbsAddLog10Name() string { return "abs_add_log10" }
