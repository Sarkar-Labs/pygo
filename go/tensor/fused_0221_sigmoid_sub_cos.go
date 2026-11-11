package tensor

import "fmt"

func Fused0221SigmoidSubCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_sub_cos: nil operand")
	}
	r := x.Sub(y)
	r = r.Sigmoid()
	r = r.Cos()
	return r, nil
}
func Fused0221SigmoidSubCosName() string { return "sigmoid_sub_cos" }
