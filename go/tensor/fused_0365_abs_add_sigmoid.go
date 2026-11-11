package tensor

import "fmt"

func Fused0365AbsAddSigmoid(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_add_sigmoid: nil operand")
	}
	r := x.Add(y)
	r = r.Abs()
	r = r.Sigmoid()
	return r, nil
}
func Fused0365AbsAddSigmoidName() string { return "abs_add_sigmoid" }
