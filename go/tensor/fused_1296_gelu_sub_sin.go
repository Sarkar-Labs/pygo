package tensor

import "fmt"

func Fused1296GeluSubSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_sub_sin: nil operand")
	}
	r := x.Sub(y)
	r = r.GELU()
	r = r.Sin()
	return r, nil
}
func Fused1296GeluSubSinName() string { return "gelu_sub_sin" }
