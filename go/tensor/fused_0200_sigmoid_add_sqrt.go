package tensor

import "fmt"

func Fused0200SigmoidAddSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_add_sqrt: nil operand")
	}
	r := x.Add(y)
	r = r.Sigmoid()
	r = r.Sqrt()
	return r, nil
}
func Fused0200SigmoidAddSqrtName() string { return "sigmoid_add_sqrt" }
