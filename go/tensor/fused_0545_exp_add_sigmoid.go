package tensor

import "fmt"

func Fused0545ExpAddSigmoid(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_add_sigmoid: nil operand")
	}
	r := x.Add(y)
	r = r.Exp()
	r = r.Sigmoid()
	return r, nil
}
func Fused0545ExpAddSigmoidName() string { return "exp_add_sigmoid" }
