package tensor

import "fmt"

func Fused0025ExpAdd(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_add: nil operand")
	}
	r := x.Add(y)
	r = r.Exp()
	return r, nil
}
func Fused0025ExpAddName() string { return "exp_add" }
