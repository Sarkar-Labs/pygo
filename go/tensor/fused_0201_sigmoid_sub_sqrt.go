package tensor

import "fmt"

func Fused0201SigmoidSubSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_sub_sqrt: nil operand")
	}
	r := x.Sub(y)
	r = r.Sigmoid()
	r = r.Sqrt()
	return r, nil
}
func Fused0201SigmoidSubSqrtName() string { return "sigmoid_sub_sqrt" }
