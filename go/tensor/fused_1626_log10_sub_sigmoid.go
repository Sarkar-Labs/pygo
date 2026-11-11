package tensor

import "fmt"

func Fused1626Log10SubSigmoid(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_sub_sigmoid: nil operand")
	}
	r := x.Sub(y)
	r = r.Log10()
	r = r.Sigmoid()
	return r, nil
}
func Fused1626Log10SubSigmoidName() string { return "log10_sub_sigmoid" }
