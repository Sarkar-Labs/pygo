package tensor

import "fmt"

func Fused1646Log10SubExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_sub_exp: nil operand")
	}
	r := x.Sub(y)
	r = r.Log10()
	r = r.Exp()
	return r, nil
}
func Fused1646Log10SubExpName() string { return "log10_sub_exp" }
