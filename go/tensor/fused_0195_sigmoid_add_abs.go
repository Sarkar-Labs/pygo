package tensor

import "fmt"

func Fused0195SigmoidAddAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_add_abs: nil operand")
	}
	r := x.Add(y)
	r = r.Sigmoid()
	r = r.Abs()
	return r, nil
}
func Fused0195SigmoidAddAbsName() string { return "sigmoid_add_abs" }
