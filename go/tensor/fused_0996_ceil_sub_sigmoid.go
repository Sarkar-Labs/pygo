package tensor

import "fmt"

func Fused0996CeilSubSigmoid(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_sub_sigmoid: nil operand")
	}
	r := x.Sub(y)
	r = r.Ceil()
	r = r.Sigmoid()
	return r, nil
}
func Fused0996CeilSubSigmoidName() string { return "ceil_sub_sigmoid" }
