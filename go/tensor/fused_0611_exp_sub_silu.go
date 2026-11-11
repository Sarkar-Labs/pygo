package tensor

import "fmt"

func Fused0611ExpSubSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_sub_silu: nil operand")
	}
	r := x.Sub(y)
	r = r.Exp()
	r = r.SiLU()
	return r, nil
}
func Fused0611ExpSubSiluName() string { return "exp_sub_silu" }
