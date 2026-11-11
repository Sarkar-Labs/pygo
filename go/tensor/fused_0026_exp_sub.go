package tensor

import "fmt"

func Fused0026ExpSub(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_sub: nil operand")
	}
	r := x.Sub(y)
	r = r.Exp()
	return r, nil
}
func Fused0026ExpSubName() string { return "exp_sub" }
