package tensor

import "fmt"

func Fused0275TanhAddSigmoid(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_add_sigmoid: nil operand")
	}
	r := x.Add(y)
	r = r.Tanh()
	r = r.Sigmoid()
	return r, nil
}
func Fused0275TanhAddSigmoidName() string { return "tanh_add_sigmoid" }
