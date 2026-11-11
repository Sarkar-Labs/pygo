package tensor

import "fmt"

func Fused0220SigmoidAddCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_add_cos: nil operand")
	}
	r := x.Add(y)
	r = r.Sigmoid()
	r = r.Cos()
	return r, nil
}
func Fused0220SigmoidAddCosName() string { return "sigmoid_add_cos" }
