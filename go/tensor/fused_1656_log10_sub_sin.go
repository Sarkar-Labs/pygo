package tensor

import "fmt"

func Fused1656Log10SubSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_sub_sin: nil operand")
	}
	r := x.Sub(y)
	r = r.Log10()
	r = r.Sin()
	return r, nil
}
func Fused1656Log10SubSinName() string { return "log10_sub_sin" }
