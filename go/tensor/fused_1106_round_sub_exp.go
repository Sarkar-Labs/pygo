package tensor

import "fmt"

func Fused1106RoundSubExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_sub_exp: nil operand")
	}
	r := x.Sub(y)
	r = r.Round()
	r = r.Exp()
	return r, nil
}
func Fused1106RoundSubExpName() string { return "round_sub_exp" }
