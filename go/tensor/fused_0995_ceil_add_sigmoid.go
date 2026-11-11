package tensor

import "fmt"

func Fused0995CeilAddSigmoid(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_add_sigmoid: nil operand")
	}
	r := x.Add(y)
	r = r.Ceil()
	r = r.Sigmoid()
	return r, nil
}
func Fused0995CeilAddSigmoidName() string { return "ceil_add_sigmoid" }
