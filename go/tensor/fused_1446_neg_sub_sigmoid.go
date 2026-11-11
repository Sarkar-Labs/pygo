package tensor

import "fmt"

func Fused1446NegSubSigmoid(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_sub_sigmoid: nil operand")
	}
	r := x.Sub(y)
	r = r.Neg()
	r = r.Sigmoid()
	return r, nil
}
func Fused1446NegSubSigmoidName() string { return "neg_sub_sigmoid" }
