package tensor

import "fmt"

func Fused0576ExpSubSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_sub_sin: nil operand")
	}
	r := x.Sub(y)
	r = r.Exp()
	r = r.Sin()
	return r, nil
}
func Fused0576ExpSubSinName() string { return "exp_sub_sin" }
