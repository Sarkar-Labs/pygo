package tensor

import "fmt"

func Fused0726SinSubSigmoid(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_sub_sigmoid: nil operand")
	}
	r := x.Sub(y)
	r = r.Sin()
	r = r.Sigmoid()
	return r, nil
}
func Fused0726SinSubSigmoidName() string { return "sin_sub_sigmoid" }
