package tensor

import "fmt"

func Fused1625Log10AddSigmoid(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_add_sigmoid: nil operand")
	}
	r := x.Add(y)
	r = r.Log10()
	r = r.Sigmoid()
	return r, nil
}
func Fused1625Log10AddSigmoidName() string { return "log10_add_sigmoid" }
