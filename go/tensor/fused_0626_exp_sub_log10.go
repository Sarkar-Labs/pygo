package tensor

import "fmt"

func Fused0626ExpSubLog10(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_sub_log10: nil operand")
	}
	r := x.Sub(y)
	r = r.Exp()
	r = r.Log10()
	return r, nil
}
func Fused0626ExpSubLog10Name() string { return "exp_sub_log10" }
