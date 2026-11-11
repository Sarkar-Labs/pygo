package tensor

import "fmt"

func Fused0205SigmoidAddExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_add_exp: nil operand")
	}
	r := x.Add(y)
	r = r.Sigmoid()
	r = r.Exp()
	return r, nil
}
func Fused0205SigmoidAddExpName() string { return "sigmoid_add_exp" }
