package tensor

import "fmt"

func Fused0625ExpAddLog10(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_add_log10: nil operand")
	}
	r := x.Add(y)
	r = r.Exp()
	r = r.Log10()
	return r, nil
}
func Fused0625ExpAddLog10Name() string { return "exp_add_log10" }
