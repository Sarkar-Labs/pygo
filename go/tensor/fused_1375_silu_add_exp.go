package tensor

import "fmt"

func Fused1375SiluAddExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_add_exp: nil operand")
	}
	r := x.Add(y)
	r = r.SiLU()
	r = r.Exp()
	return r, nil
}
func Fused1375SiluAddExpName() string { return "silu_add_exp" }
