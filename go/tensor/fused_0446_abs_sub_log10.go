package tensor

import "fmt"

func Fused0446AbsSubLog10(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_sub_log10: nil operand")
	}
	r := x.Sub(y)
	r = r.Abs()
	r = r.Log10()
	return r, nil
}
func Fused0446AbsSubLog10Name() string { return "abs_sub_log10" }
