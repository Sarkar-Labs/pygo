package tensor

import "fmt"

func Fused0206SigmoidSubExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_sub_exp: nil operand")
	}
	r := x.Sub(y)
	r = r.Sigmoid()
	r = r.Exp()
	return r, nil
}
func Fused0206SigmoidSubExpName() string { return "sigmoid_sub_exp" }
