package tensor

import "fmt"

func Fused1266GeluSubSigmoid(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_sub_sigmoid: nil operand")
	}
	r := x.Sub(y)
	r = r.GELU()
	r = r.Sigmoid()
	return r, nil
}
func Fused1266GeluSubSigmoidName() string { return "gelu_sub_sigmoid" }
