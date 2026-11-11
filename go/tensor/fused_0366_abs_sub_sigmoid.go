package tensor

import "fmt"

func Fused0366AbsSubSigmoid(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_sub_sigmoid: nil operand")
	}
	r := x.Sub(y)
	r = r.Abs()
	r = r.Sigmoid()
	return r, nil
}
func Fused0366AbsSubSigmoidName() string { return "abs_sub_sigmoid" }
