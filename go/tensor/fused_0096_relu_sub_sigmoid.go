package tensor

import "fmt"

func Fused0096ReluSubSigmoid(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_sub_sigmoid: nil operand")
	}
	r := x.Sub(y)
	r = r.ReLU()
	r = r.Sigmoid()
	return r, nil
}
func Fused0096ReluSubSigmoidName() string { return "relu_sub_sigmoid" }
