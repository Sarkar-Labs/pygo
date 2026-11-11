package tensor

import "fmt"

func Fused1645Log10AddExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_add_exp: nil operand")
	}
	r := x.Add(y)
	r = r.Log10()
	r = r.Exp()
	return r, nil
}
func Fused1645Log10AddExpName() string { return "log10_add_exp" }
