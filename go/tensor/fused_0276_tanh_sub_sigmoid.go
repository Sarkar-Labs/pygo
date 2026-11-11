package tensor

import "fmt"

func Fused0276TanhSubSigmoid(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_sub_sigmoid: nil operand")
	}
	r := x.Sub(y)
	r = r.Tanh()
	r = r.Sigmoid()
	return r, nil
}
func Fused0276TanhSubSigmoidName() string { return "tanh_sub_sigmoid" }
