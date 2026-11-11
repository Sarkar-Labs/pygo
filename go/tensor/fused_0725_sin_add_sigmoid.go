package tensor

import "fmt"

func Fused0725SinAddSigmoid(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_add_sigmoid: nil operand")
	}
	r := x.Add(y)
	r = r.Sin()
	r = r.Sigmoid()
	return r, nil
}
func Fused0725SinAddSigmoidName() string { return "sin_add_sigmoid" }
