package tensor

import "fmt"

func Fused1376SiluSubExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_sub_exp: nil operand")
	}
	r := x.Sub(y)
	r = r.SiLU()
	r = r.Exp()
	return r, nil
}
func Fused1376SiluSubExpName() string { return "silu_sub_exp" }
