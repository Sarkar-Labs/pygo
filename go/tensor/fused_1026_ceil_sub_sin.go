package tensor

import "fmt"

func Fused1026CeilSubSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_sub_sin: nil operand")
	}
	r := x.Sub(y)
	r = r.Ceil()
	r = r.Sin()
	return r, nil
}
func Fused1026CeilSubSinName() string { return "ceil_sub_sin" }
