package tensor

import "fmt"

func Fused1025CeilAddSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_add_sin: nil operand")
	}
	r := x.Add(y)
	r = r.Ceil()
	r = r.Sin()
	return r, nil
}
func Fused1025CeilAddSinName() string { return "ceil_add_sin" }
