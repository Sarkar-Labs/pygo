package tensor

import "fmt"

func Fused1176SignSubSigmoid(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_sub_sigmoid: nil operand")
	}
	r := x.Sub(y)
	r = r.Sign()
	r = r.Sigmoid()
	return r, nil
}
func Fused1176SignSubSigmoidName() string { return "sign_sub_sigmoid" }
