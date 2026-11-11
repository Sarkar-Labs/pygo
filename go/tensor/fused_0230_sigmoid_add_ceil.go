package tensor

import "fmt"

func Fused0230SigmoidAddCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_add_ceil: nil operand")
	}
	r := x.Add(y)
	r = r.Sigmoid()
	r = r.Ceil()
	return r, nil
}
func Fused0230SigmoidAddCeilName() string { return "sigmoid_add_ceil" }
