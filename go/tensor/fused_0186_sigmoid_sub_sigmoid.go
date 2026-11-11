package tensor

import "fmt"

func Fused0186SigmoidSubSigmoid(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_sub_sigmoid: nil operand")
	}
	r := x.Sub(y)
	r = r.Sigmoid()
	r = r.Sigmoid()
	return r, nil
}
func Fused0186SigmoidSubSigmoidName() string { return "sigmoid_sub_sigmoid" }
