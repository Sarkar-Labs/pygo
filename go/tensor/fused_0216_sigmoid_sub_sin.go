package tensor

import "fmt"

func Fused0216SigmoidSubSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_sub_sin: nil operand")
	}
	r := x.Sub(y)
	r = r.Sigmoid()
	r = r.Sin()
	return r, nil
}
func Fused0216SigmoidSubSinName() string { return "sigmoid_sub_sin" }
