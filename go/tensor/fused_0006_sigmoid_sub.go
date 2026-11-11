package tensor

import "fmt"

func Fused0006SigmoidSub(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_sub: nil operand")
	}
	r := x.Sub(y)
	r = r.Sigmoid()
	return r, nil
}
func Fused0006SigmoidSubName() string { return "sigmoid_sub" }
