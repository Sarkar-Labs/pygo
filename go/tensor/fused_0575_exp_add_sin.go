package tensor

import "fmt"

func Fused0575ExpAddSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_add_sin: nil operand")
	}
	r := x.Add(y)
	r = r.Exp()
	r = r.Sin()
	return r, nil
}
func Fused0575ExpAddSinName() string { return "exp_add_sin" }
