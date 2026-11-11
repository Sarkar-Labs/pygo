package tensor

import "fmt"

func Fused1265GeluAddSigmoid(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_add_sigmoid: nil operand")
	}
	r := x.Add(y)
	r = r.GELU()
	r = r.Sigmoid()
	return r, nil
}
func Fused1265GeluAddSigmoidName() string { return "gelu_add_sigmoid" }
