package tensor

import "fmt"

func Fused0546ExpSubSigmoid(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_sub_sigmoid: nil operand")
	}
	r := x.Sub(y)
	r = r.Exp()
	r = r.Sigmoid()
	return r, nil
}
func Fused0546ExpSubSigmoidName() string { return "exp_sub_sigmoid" }
